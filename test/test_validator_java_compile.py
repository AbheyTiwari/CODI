import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from core.validator import Validator
from state.temp_db import RunState, ToolResult


class ValidatorJavaCompileTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("mvn"), "mvn is not on PATH")
    def test_java_compile_failure_blocks_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            source = project / "src" / "main" / "java" / "App.java"
            source.parent.mkdir(parents=True)
            source.write_text(
                "\n".join([
                    "public class App {",
                    "    public static void main(String[] args) throws MissingException {",
                    "        throw new MissingException();",
                    "    }",
                    "}",
                ]),
                encoding="utf-8",
            )
            (project / "pom.xml").write_text(
                "\n".join([
                    '<project xmlns="http://maven.apache.org/POM/4.0.0"',
                    '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
                    '         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">',
                    "    <modelVersion>4.0.0</modelVersion>",
                    "    <groupId>codi.test</groupId>",
                    "    <artifactId>compile-check</artifactId>",
                    "    <version>1.0-SNAPSHOT</version>",
                    "    <properties>",
                    "        <maven.compiler.source>11</maven.compiler.source>",
                    "        <maven.compiler.target>11</maven.compiler.target>",
                    "    </properties>",
                    "</project>",
                ]),
                encoding="utf-8",
            )

            old_working_dir = os.environ.get("CODI_WORKING_DIR")
            os.environ["CODI_WORKING_DIR"] = str(project)
            try:
                state = RunState()
                state.tool_results = [
                    ToolResult(
                        "write_file",
                        "ok",
                        json.dumps({"success": True, "file_modified": str(source)}),
                    )
                ]

                result = Validator().validate(state)

                self.assertFalse(result)
                self.assertIn("JAVA COMPILE FAILED", state.validation_notes)
            finally:
                if old_working_dir is None:
                    os.environ.pop("CODI_WORKING_DIR", None)
                else:
                    os.environ["CODI_WORKING_DIR"] = old_working_dir


if __name__ == "__main__":
    unittest.main()
