import json
import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_compose(relative_path):
    return yaml.safe_load((ROOT / relative_path).read_text(encoding="utf-8-sig"))


class ComposeSecurityTests(unittest.TestCase):
    def test_grafana_has_loopback_binding_and_explicit_environment(self):
        services = load_compose("grafana/docker-compose.yml")["services"]
        grafana = services["grafana"]
        self.assertEqual(grafana["ports"], ["127.0.0.1:3000:3000"])
        self.assertNotIn("env_file", grafana)
        self.assertEqual(grafana["entrypoint"], ["/bin/sh", "-eu", "-c"])
        preflight = grafana["command"][0]
        self.assertIn('[ "$$GF_SECURITY_ADMIN_PASSWORD" = "admin" ]', preflight)
        self.assertIn("Grafana refuses the vendor-default admin password", preflight)
        self.assertIn("exec /run.sh", preflight)
        self.assertIn("${GF_SECURITY_ADMIN_USER:?", grafana["environment"]["GF_SECURITY_ADMIN_USER"])
        self.assertIn("${GF_SECURITY_ADMIN_PASSWORD:?", grafana["environment"]["GF_SECURITY_ADMIN_PASSWORD"])
        example = (ROOT / "notification" / "grafana.env.example").read_text(encoding="utf-8")
        example_keys = {
            line.split("=", 1)[0]
            for line in example.splitlines()
            if line and "=" in line
        }
        required_interpolations = {
            match.group(1)
            for value in grafana["environment"].values()
            if isinstance(value, str)
            for match in re.finditer(r"\$\{([A-Za-z_][A-Za-z0-9_]*):\?", value)
        }
        self.assertTrue(required_interpolations)
        self.assertTrue(required_interpolations.issubset(example_keys))
        self.assertEqual(example_keys, required_interpolations)
        self.assertEqual(
            set(grafana["environment"]),
            {
                "GF_SECURITY_ADMIN_USER",
                "GF_SECURITY_ADMIN_PASSWORD",
                "GF_USERS_ALLOW_SIGN_UP",
                "INFLUX_URL",
                "ELASTIC_URL",
                "INFLUX_ORG",
                "INFLUX_BUCKET",
                "INFLUX_TOKEN",
                "ELASTIC_ALERT_INDEX",
                "ELASTIC_READER_API_KEY",
            },
        )

    def test_kafka_host_ports_are_loopback_and_internal_listener_remains(self):
        services = load_compose("grafana/docker-compose.yml")["services"]
        self.assertEqual(services["kafka-broker"]["ports"], ["127.0.0.1:9092:9092"])
        self.assertEqual(services["kafka-ui"]["ports"], ["127.0.0.1:8083:8080"])
        self.assertIn("PLAINTEXT://kafka-broker:29092", services["kafka-broker"]["environment"]["KAFKA_ADVERTISED_LISTENERS"])
        self.assertEqual(services["kafka-ui"]["environment"]["KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS"], "kafka-broker:29092")

    def test_only_opencti_platform_receives_admin_token(self):
        services = load_compose("opencti/docker-compose.yml")["services"]
        admin_token_services = [
            name
            for name, service in services.items()
            if "OPENCTI_ADMIN_TOKEN" in str(service.get("environment", {}))
        ]
        self.assertEqual(admin_token_services, ["opencti-platform"])
        expected_tokens = {
            "opencti-worker": "OPENCTI_WORKER_ADMIN_TOKEN",
            "connector-mitre": "OPENCTI_MITRE_CONNECTOR_TOKEN",
            "connector-alienvault": "OPENCTI_ALIENVAULT_CONNECTOR_TOKEN",
            "connector-abuse-ssl": "OPENCTI_ABUSE_SSL_CONNECTOR_TOKEN",
            "connector-urlhaus": "OPENCTI_URLHAUS_CONNECTOR_TOKEN",
            "connector-virustotal-livehunt": "OPENCTI_VIRUSTOTAL_LIVEHUNT_CONNECTOR_TOKEN",
            "connector-virustotal": "OPENCTI_VIRUSTOTAL_CONNECTOR_TOKEN",
            "connector-abuseipdb": "OPENCTI_ABUSEIPDB_CONNECTOR_TOKEN",
            "connector-ipinfo": "OPENCTI_IPINFO_CONNECTOR_TOKEN",
            "connector-shodan": "OPENCTI_SHODAN_CONNECTOR_TOKEN",
        }
        token_services = {
            name
            for name, service in services.items()
            if "OPENCTI_TOKEN" in service.get("environment", {})
        }
        self.assertEqual(token_services, set(expected_tokens))
        mapped_tokens = set()
        for name, variable in expected_tokens.items():
            token = services[name]["environment"]["OPENCTI_TOKEN"]
            self.assertIn(f"${{{variable}:?", token, name)
            mapped_tokens.add(variable)
        self.assertEqual(len(mapped_tokens), len(expected_tokens))
        example = (ROOT / "opencti" / ".env.example").read_text(encoding="utf-8")
        for variable in ("OPENCTI_ADMIN_TOKEN", *expected_tokens.values()):
            self.assertIn(f"{variable}=", example)

    def test_influx_initialization_variables_fail_when_missing(self):
        environment = load_compose("observability/docker-compose.yml")["services"]["influxdb"]["environment"]
        for key in (
            "DOCKER_INFLUXDB_INIT_USERNAME",
            "DOCKER_INFLUXDB_INIT_PASSWORD",
            "DOCKER_INFLUXDB_INIT_ADMIN_TOKEN",
        ):
            self.assertIn(":?", environment[key])
        example = (ROOT / "observability" / ".env.example").read_text(encoding="utf-8")
        self.assertIn("INFLUX_INIT_USERNAME=", example)
        self.assertIn("INFLUX_INIT_PASSWORD=", example)
        self.assertIn("INFLUX_INIT_ADMIN_TOKEN=", example)

    def test_faithfulness_panel_states_it_is_offline_and_charts_nothing(self):
        """M2 exists now, but only as an offline batch over a fixed 100-alert sample.

        This test used to require the title to say "Paused" and the body to say no metric
        is produced. Both became false once M2 was measured, so the assertions move to the
        part that is still load-bearing: the dispatcher writes no faithfulness series, so
        the panel must stay a text notice with no query. A chart here would be reading a
        field that is never written, which renders as an empty graph that looks like "no
        alerts" rather than "not measured this way".
        """
        dashboard = json.loads(
            (ROOT / "grafana" / "dashboards" / "soc-overview.json").read_text(encoding="utf-8")
        )
        panel = next(item for item in dashboard["panels"] if item["type"] == "text")
        self.assertIn("Faithfulness", panel["title"])
        self.assertIn("offline", panel["title"].lower() + panel["options"]["content"].lower())
        self.assertNotIn("targets", panel)
        self.assertNotIn("soc_llm_eval_summary", json.dumps(panel))

        # and no other panel may invent a faithfulness series either
        for other in dashboard["panels"]:
            for target in other.get("targets", []):
                self.assertNotIn("faithfulness", target.get("query", "").lower())


if __name__ == "__main__":
    unittest.main()
