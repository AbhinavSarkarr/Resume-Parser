import streamlit as st
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from typing import List
import pdfplumber
import fitz
import pytesseract
from PIL import Image
import io
import os
from dotenv import dotenv_values, load_dotenv
import json

# loading env variables
config = dotenv_values(".env")
load_dotenv()

# Function to extract text from the resume
def extract_text_from_resume(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        lp_text = ""
        for i in range(len(pdf.pages)):
            lp_text += '\n' + pdf.pages[i].extract_text()
    lp_text = lp_text.strip()
    lp_words = lp_text.split()
    
    if len(lp_words) >= 50:
        return lp_text

    document = fitz.open(pdf_path)
    text = ""
    for page in document:
        text += page.get_text()
    document.close()
    pdf_words = text.split()

    if len(pdf_words) >= 50:
        return text.strip()

    with fitz.open(pdf_path) as doc:
        all_text = ''
        for page in doc:
            image_list = page.get_images(full=True)
            for image_index, img in enumerate(page.get_images(full=True)):
                base_image = doc.extract_image(img[0])
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                ocr_text = pytesseract.image_to_string(image, lang='eng')
                all_text += ocr_text + '\n'

    ocr_words = all_text.split()

    if len(ocr_words) >= 50:
        return all_text
    return None

# Streamlit UI
st.title("Resume Parsing Application")

# Upload Resume
uploaded_file = st.file_uploader("Choose a resume PDF file", type="pdf")

# If a file is uploaded
if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("uploaded_resume.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text from the resume
    extracted_text = extract_text_from_resume("uploaded_resume.pdf")
    
    if extracted_text:
        # Define the Pydantic model
        class ResumeOutput(BaseModel):
            name: str
            highest_qulification: str
            contact_details: str
            job_role: str
            suitable_job_designations: List[str]
            preferred_job_locations: List[str]
            experience_level: str
            years_of_experience: str
            profile_summary: str
            cover_letter: str
            possible_set_of_general_questions_according_to_the_resume: dict

        # Create the parser using the Pydantic model
        output_parser = PydanticOutputParser(pydantic_object=ResumeOutput)

        # Declare the LLM
        llm = ChatGroq(
            temperature=0.0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192",
        )

        # Define the chain with the Pydantic parser
        chain = ChatPromptTemplate.from_template("""
            You are a Resume Parser Tool.
            Content: {content}
            Extract the below information using the provided content:
                1. Name
                2. Highest Qualification
                3. Address(Hometown or the place where he/she is staying)
                4. Contact Details
                5. Job Role
                6. Suitable Job Designations according to the skill sets
                7. Preferred Job Locations
                8. Experience level the candidate has
                9. Years of Experience
                10. Provide a profile summary of the person whose resume is provided in 500 words
                11. Cover Letter Snippet in 800 words according to the context
                12. A possible set(atleast 20) of interview questions along with their answers that may be asked on the basis of the resume

            Present the information in JSON format with the following structure:
            {{
                "name": "Full Name",
                "highest_qulification": "Qualification",
                "address": "Address",
                "contact_details": "number/mail",
                "job_role": "Internship or Full time",
                "suitable_job_designations": ["Job designations 1", "Job designations 2"],
                "preferred_job_locations": ["Location 1", "Location 2"],
                "experience_level": "Experience Level",
                "years_of_experience": "number of years working",
                "profile_summary": "Summary text",
                "cover_letter": "cover_letter according to the context",
                "possible_set_of_general_questions_according_to_the_resume": {{"question":"answer", "question":"answer"}}
            }}
        """) | llm | output_parser

        # Run the chain on the extracted text
        parsed_output = chain.invoke({"content": extracted_text})

        # Convert parsed output to dictionary
        parsed_output_dict = parsed_output.dict()

        # Display the parsed information in a dashboard format
        st.header("Resume Summary")

        st.subheader("Personal Details")
        st.markdown(f"**Name:** {parsed_output_dict['name']}")
        st.markdown(f"**Highest Qualification:** {parsed_output_dict['highest_qulification']}")
        st.markdown(f"**Contact Details:** {parsed_output_dict['contact_details']}")

        st.subheader("Job Preferences")
        st.markdown(f"**Job Role:** {parsed_output_dict['job_role']}")
        st.markdown(f"**Suitable Job Designations:** {', '.join(parsed_output_dict['suitable_job_designations'])}")
        st.markdown(f"**Preferred Job Locations:** {', '.join(parsed_output_dict['preferred_job_locations'])}")
        st.markdown(f"**Experience Level:** {parsed_output_dict['experience_level']}")
        st.markdown(f"**Years of Experience:** {parsed_output_dict['years_of_experience']}")

        st.subheader("Profile Summary")
        st.markdown(parsed_output_dict['profile_summary'])

        st.subheader("Cover Letter")
        st.markdown(parsed_output_dict['cover_letter'])

        st.subheader("Possible Interview Questions")
        for question, answer in parsed_output_dict['possible_set_of_general_questions_according_to_the_resume'].items():
            st.markdown(f"**Q: {question}**")
            st.markdown(f"**A:** {answer}")
    else:
        st.error("Could not extract enough text from the resume.")
else:
    st.info("Please upload a PDF resume file to extract information.")
