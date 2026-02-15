import os
from datetime import datetime

# ==========================================================
# PDF REPORT GENERATION (Feature 7)
# Generates forensic reports in PDF format
# Uses fpdf2 library (pip install fpdf2)
# ==========================================================

def generate_report(results, output_path=None):
    """
    Generate a PDF forensic report from analysis results.
    
    Args:
        results: dict containing analysis data from any detector
        output_path: Path to save PDF (default: auto-generated)
        
    Returns:
        str: Path to generated PDF
    """
    try:
        from fpdf import FPDF
    except ImportError:
        print("⚠️  fpdf2 not installed. Generating text report instead.")
        return generate_text_report(results, output_path)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"forensic_report_{timestamp}.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # === HEADER ===
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "AI FORENSIC REPORT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)

    # === SEPARATOR ===
    pdf.set_draw_color(0, 120, 200)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # === FILE INFO ===
    if "file" in results or "folder" in results:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "1. File Information", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        
        if "file" in results:
            pdf.cell(0, 6, f"File: {results.get('file', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
        if "folder" in results:
            pdf.cell(0, 6, f"Folder: {results.get('folder', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
        if "total_files" in results:
            pdf.cell(0, 6, f"Total Files: {results['total_files']}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

    # === VERDICT ===
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Verdict", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)

    is_fake = results.get("is_fake", results.get("is_video_fake", None))
    confidence = results.get("confidence", results.get("final_confidence", 0))
    label = results.get("label", results.get("final_label", ""))

    if is_fake is True:
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 8, "VERDICT: AI GENERATED / FAKE", new_x="LMARGIN", new_y="NEXT")
    elif is_fake is False:
        pdf.set_text_color(0, 150, 0)
        pdf.cell(0, 8, "VERDICT: AUTHENTIC / REAL", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_text_color(200, 150, 0)
        pdf.cell(0, 8, f"VERDICT: {label or 'INCONCLUSIVE'}", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 6, f"Confidence: {confidence*100:.2f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # === MODEL RESULTS (Ensemble) ===
    if "model_results" in results:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "3. Model Breakdown", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)

        # Table header
        pdf.set_fill_color(230, 230, 250)
        pdf.cell(50, 7, "Model", border=1, fill=True)
        pdf.cell(50, 7, "Prediction", border=1, fill=True)
        pdf.cell(40, 7, "Confidence", border=1, fill=True)
        pdf.cell(30, 7, "Weight", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

        for mr in results["model_results"]:
            pdf.cell(50, 7, mr["model"], border=1)
            pdf.cell(50, 7, "AI" if "AI" in mr.get("label", "") else "REAL", border=1)
            pdf.cell(40, 7, f"{mr['confidence']*100:.1f}%", border=1)
            pdf.cell(30, 7, f"{mr.get('vote_weight', 1.0):.1f}", border=1, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

    # === BATCH RESULTS ===
    if "results" in results and isinstance(results["results"], list):
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "3. Batch Results", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)

        pdf.set_fill_color(230, 230, 250)
        pdf.cell(80, 7, "File", border=1, fill=True)
        pdf.cell(30, 7, "Type", border=1, fill=True)
        pdf.cell(40, 7, "Result", border=1, fill=True)
        pdf.cell(30, 7, "Confidence", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

        for r in results["results"][:50]:  # Limit to 50 rows
            fname = os.path.basename(r["file"])[:35]
            pdf.cell(80, 6, fname, border=1)
            pdf.cell(30, 6, r["type"], border=1)
            pdf.cell(40, 6, "FAKE" if r["is_fake"] else "REAL", border=1)
            pdf.cell(30, 6, f"{r['confidence']*100:.1f}%", border=1, new_x="LMARGIN", new_y="NEXT")
        
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, f"Summary: {results.get('fake_count', 0)} Fake / {results.get('real_count', 0)} Real / {results.get('error_count', 0)} Errors", new_x="LMARGIN", new_y="NEXT")

    # === METADATA ===
    if "metadata" in results:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "4. Metadata Analysis", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)

        meta = results["metadata"]
        if "camera_info" in meta and meta["camera_info"]:
            pdf.cell(0, 6, f"Camera: {meta['camera_info']}", new_x="LMARGIN", new_y="NEXT")
        if "software_info" in meta and meta["software_info"]:
            pdf.cell(0, 6, f"Software: {meta['software_info']}", new_x="LMARGIN", new_y="NEXT")

    # === FINDINGS ===
    if "findings" in results and results["findings"]:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "5. Forensic Findings", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)

        for f in results["findings"]:
            severity = f.get("severity", "INFO")
            if severity == "HIGH":
                pdf.set_text_color(200, 0, 0)
            elif severity == "MEDIUM":
                pdf.set_text_color(200, 150, 0)
            else:
                pdf.set_text_color(0, 0, 200)
            
            pdf.cell(0, 6, f"[{severity}] {f['finding']}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 5, f"   {f.get('detail', '')}", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_text_color(0, 0, 0)

    # === FOOTER ===
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 5, "Generated by C2PA AI Detection System", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 5, "This report is for informational purposes only", new_x="LMARGIN", new_y="NEXT", align="C")

    # Save
    pdf.output(output_path)
    print(f"\n📄 Report saved: {output_path}")
    return output_path


def generate_text_report(results, output_path=None):
    """
    Fallback: Generate a text report if fpdf2 is not installed.
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"forensic_report_{timestamp}.txt"

    lines = []
    lines.append("=" * 60)
    lines.append("AI FORENSIC REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")

    is_fake = results.get("is_fake", results.get("is_video_fake", None))
    confidence = results.get("confidence", results.get("final_confidence", 0))

    if is_fake is True:
        lines.append("VERDICT: AI GENERATED / FAKE")
    elif is_fake is False:
        lines.append("VERDICT: AUTHENTIC / REAL")
    else:
        lines.append(f"VERDICT: {results.get('label', 'INCONCLUSIVE')}")

    lines.append(f"Confidence: {confidence*100:.2f}%")
    lines.append("")

    if "model_results" in results:
        lines.append("MODEL BREAKDOWN:")
        for mr in results["model_results"]:
            lines.append(f"  {mr['model']:15s} → {'AI' if 'AI' in mr.get('label', '') else 'REAL'} ({mr['confidence']*100:.1f}%)")

    if "findings" in results:
        lines.append("\nFINDINGS:")
        for f in results["findings"]:
            lines.append(f"  [{f['severity']}] {f['finding']}")

    if "results" in results and isinstance(results["results"], list):
        lines.append(f"\nBATCH RESULTS ({len(results['results'])} files):")
        for r in results["results"]:
            fname = os.path.basename(r["file"])
            lines.append(f"  {fname:40s} {'FAKE' if r['is_fake'] else 'REAL':8s} {r['confidence']*100:.1f}%")

    lines.append("")
    lines.append("=" * 60)

    report_text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n📄 Report saved: {output_path}")
    return output_path


# ==========================================================
# CLI ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    # Demo with sample data
    sample = {
        "file": "test_image.jpg",
        "is_fake": True,
        "confidence": 0.87,
        "final_label": "AI GENERATED",
        "model_results": [
            {"model": "CNN3", "label": "AI", "confidence": 0.82, "vote_weight": 1.0},
            {"model": "HybridFFT", "label": "AI", "confidence": 0.95, "vote_weight": 2.0},
        ],
        "findings": [
            {"severity": "HIGH", "finding": "No camera info", "detail": "Missing EXIF"},
        ]
    }
    generate_report(sample)
