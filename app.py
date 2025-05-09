import os
import subprocess
import glob
import streamlit as st

# Sidebar: language selection and file uploader
st.sidebar.title("Configuration")
lang = st.sidebar.selectbox("Select language code", ["ta", "en"])
uploaded_file = st.sidebar.file_uploader("Upload file for summarization", type=["txt", "json", "csv", "docx"])

# Define directory paths
INPUT_DIR = os.path.join("test_input", lang)
INTERMED_DIR = os.path.join("test_intermed", lang)
PREP_DIR = os.path.join("test_prep_path", lang)
OUTPUT_DIR = os.path.join("test_output", lang)

# Create directories if they don't exist
for path in [INPUT_DIR, INTERMED_DIR, PREP_DIR, OUTPUT_DIR]:
    os.makedirs(path, exist_ok=True)

# Main processing
if uploaded_file:
    # Save uploaded file
    input_path = os.path.join(INPUT_DIR, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Saved file to {input_path}")

    if st.sidebar.button("Run Pipeline"):
        st.title("Pipeline Progress")
        progress = st.progress(0)
        status_text = st.empty()

        # Step 1: initiateSummarisation
        status_text.text("Step 1/3: Running initiateSummarisation.py...")
        cmd1 = ["python", "initiateSummarisation.py",
                "--input_dir", INPUT_DIR,
                "--output_dir", INTERMED_DIR,
                "--lang", lang]
        subprocess.run(cmd1, check=True)
        progress.progress(33)

        # Step 2: prepareData
        status_text.text("Step 2/3: Running prepareData.py...")
        cmd2 = ["python", "prepareData.py",
                "--data_path", INTERMED_DIR,
                "--prep_path", PREP_DIR,
                "--lang", lang]
        subprocess.run(cmd2, check=True)
        progress.progress(66)

        # Step 3: legal_ilp_best
        status_text.text("Step 3/3: Running legal_ilp_best.py...")
        cmd3 = ["python", "legal_ilp_best.py",
                "--json_path", os.path.join(PREP_DIR, "prepared_data.json"),
                "--length_file", "length_file.txt",
                "--output_dir", OUTPUT_DIR,
                "--lang", lang]
        subprocess.run(cmd3, check=True)
        progress.progress(100)

        status_text.text("Processing complete.")

        # Display result file
        result_files = glob.glob(os.path.join(OUTPUT_DIR, "*"))
        if result_files:
            result_file = result_files[0]
            st.subheader("Result File")
            try:
                with open(result_file, "r", encoding="utf-8") as rf:
                    content = rf.read()
                st.code(content)
            except Exception:
                st.write(f"Result file available at: {result_file}")
        else:
            st.error("No result file found.")
