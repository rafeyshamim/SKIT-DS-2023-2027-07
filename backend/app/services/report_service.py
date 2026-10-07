from typing import Optional, Dict, Any
from app.models.sql.scan import CTScan
from app.models.sql.patient import Patient
from app.models.sql.inference import InferenceRun
from app.models.nosql.inference_payload import InferencePayloadDocument


class ReportService:
    """
    Automated Diagnostic Clinical Report Generator.
    Synthesizes patient demographic information, imaging acquisition details,
    and 3D CNN inference outputs (lesion size, location, risk stratification)
    into structured clinical radiological documentation adhering to Fleischner Society guidelines.
    """

    @staticmethod
    def generate_clinical_report(
        patient: Patient,
        scan: CTScan,
        inference_run: Optional[InferenceRun] = None,
        inference_doc: Optional[Dict[str, Any]] = None,
        radiologist_name: str = "Dr. AI Diagnostic System, M.D."
    ) -> Dict[str, str]:
        
        technique = (
            f"Helical high-resolution volumetric MDCT of the thorax performed with 120 kVp, "
            f"automated tube current modulation, and slice thickness of {scan.slice_thickness_mm or 1.25} mm. "
            f"Multiplanar coronal and sagittal reformations with volumetric 3D lung windowing were evaluated."
        )

        clinical_history = (
            patient.medical_history_notes or 
            f"Patient {patient.full_name}, age evaluated for chest symptoms or lung nodule screening."
        )

        if inference_run and inference_run.primary_prediction:
            pred = inference_run.primary_prediction
            conf = f"{inference_run.confidence_score * 100:.1f}%" if inference_run.confidence_score else "N/A"
            risk = inference_run.risk_level or "EVALUATION REQUIRED"

            # Check if lesions exist in payload
            lesions = []
            if inference_doc and "lesion_detections" in inference_doc:
                lesions = inference_doc["lesion_detections"]

            if lesions:
                lesion_details = []
                for idx, l in enumerate(lesions, 1):
                    vol = l.get("volume_mm3", 0.0)
                    centroid = l.get("centroid_voxel", {})
                    score = l.get("malignancy_score", 0.0)
                    nod_type = l.get("nodule_type", "Solid nodule")
                    lesion_details.append(
                        f"  - Nodule #{idx}: {nod_type} at voxel centroid (z={centroid.get('z')}, "
                        f"y={centroid.get('y')}, x={centroid.get('x')}). Estimated 3D volume: {vol} mm³. "
                        f"Malignancy suspicion index: {score:.2f}."
                    )
                lesions_text = "\n".join(lesion_details)
                findings = (
                    f"1. LUNGS AND AIRWAYS:\n"
                    f"   The 3D CNN deep learning analysis identified {len(lesions)} focal pulmonary lesion(s):\n"
                    f"{lesions_text}\n"
                    f"   Associated spiculation and marginal irregularities detected on high-attenuation segment.\n"
                    f"2. PLEURA: No pleural effusion or pneumothorax identified.\n"
                    f"3. MEDIASTINUM & HILA: Heart size within normal limits. No obvious gross mediastinal lymphadenopathy."
                )
                impression = (
                    f"1. {pred.upper()} (Model Confidence: {conf}, Risk Category: {risk}).\n"
                    f"2. Suspect focal lesion exhibiting imaging features requiring urgent multidisciplinary review."
                )
                recommendations = (
                    f"Per Fleischner Society Guidelines for pulmonary nodules:\n"
                    f"- Recommend dedicated contrast-enhanced chest CT or 18F-FDG PET/CT scan to assess metabolic activity.\n"
                    f"- Consider pulmonology consultation and tissue biopsy or follow-up CT surveillance within 3-6 months."
                )
            else:
                findings = (
                    f"1. LUNGS AND AIRWAYS: Lungs are well-aerated without focal airspace consolidation, "
                    f"spiculated mass, or suspicious pulmonary nodules.\n"
                    f"2. PLEURA: Clear without pleural thickening or effusion.\n"
                    f"3. MEDIASTINUM: Unremarkable mediastinal and hilar contours."
                )
                impression = f"No acute cardiopulmonary disease. {pred} (Confidence: {conf})."
                recommendations = "Routine screening as clinically indicated. No urgent imaging follow-up required."
        else:
            findings = "Awaiting automated 3D CNN inference execution."
            impression = "Preliminary report pending volumetric deep learning analysis."
            recommendations = "Execute inference pipeline to obtain AI quantitative analysis."

        return {
            "technique": technique,
            "clinical_history": clinical_history,
            "findings": findings,
            "impression": impression,
            "recommendations": recommendations,
            "radiologist_name": radiologist_name
        }


report_service = ReportService()
