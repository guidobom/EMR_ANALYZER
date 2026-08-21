from pathlib import Path
import json
import tempfile
import unittest
import zipfile

from emr_analyzer.pipeline.pdf_extractor import PdfPlumberExtractor
from emr_analyzer.utils.file_utils import (
    detect_file_type,
    is_supported_file,
    supported_file_dialog_filter,
)


class InputFormatsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.extractor = PdfPlumberExtractor()

    def tearDown(self):
        self.tmp.cleanup()

    def test_text_csv_xml_json_and_docx_are_deterministic(self):
        text = self.root / "referto.hl7"
        text.write_text("MSH|^~\\&|LAB\rOBX|1|NM|TSH||0.01|mIU/L", encoding="utf-8")
        csv_path = self.root / "lab.csv"
        csv_path.write_text("Parametro;Valore\nTSH;0.01", encoding="utf-8")
        xml_path = self.root / "cda.xml"
        xml_path.write_text("<ClinicalDocument><text>Dispnea</text></ClinicalDocument>", encoding="utf-8")
        json_path = self.root / "fhir.json"
        json_path.write_text(json.dumps({"resourceType": "Observation", "value": 7}), encoding="utf-8")
        docx_path = self.root / "nota.docx"
        with zipfile.ZipFile(docx_path, "w") as archive:
            archive.writestr(
                "word/document.xml",
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                '<w:body><w:p><w:r><w:t>Tosse persistente</w:t></w:r></w:p></w:body></w:document>',
            )

        self.assertIn("OBX", self.extractor.convert(text).plain_text)
        self.assertEqual(self.extractor.convert(csv_path).tables[0].rows[1][0], "TSH")
        self.assertIn("Dispnea", self.extractor.convert(xml_path).plain_text)
        self.assertIn("resourceType: Observation", self.extractor.convert(json_path).plain_text)
        self.assertIn("Tosse persistente", self.extractor.convert(docx_path).plain_text)
        for path in (text, csv_path, xml_path, json_path, docx_path):
            self.assertTrue(is_supported_file(path))
            self.assertNotEqual(detect_file_type(path), "unsupported")

    def test_file_dialog_exposes_every_supported_family(self):
        file_filter = supported_file_dialog_filter()
        for extension in (".pdf", ".tiff", ".hl7", ".cda", ".docx", ".xlsx"):
            self.assertIn(f"*{extension}", file_filter)


if __name__ == "__main__":
    unittest.main()
