import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

# Initialize the Gemini Client
client = genai.Client()

def grade_response(student_response: str, rubric: str, question: str) -> str:
    """
    Sends the student response and grading criteria to Gemini 2.5 Flash
    to evaluate and return a graded breakdown.
    """
    print("Sending student work to Gemini for evaluation...")
    
    # Constructing a structured prompt for the model
    prompt = f"""
You are an expert, precise academic grading assistant. Grade the following student response based strictly on the provided question and rubric.

### Context
Question/Problem:
{question}

Grading Rubric:
{rubric}

### Student Submission
{student_response}

### Instructions
1. Analyze the student's process step-by-step.
2. Award partial credit clearly based on the rubric criteria.
3. Call out any conceptual misunderstandings or mathematical errors.
4. Provide the final point total at the very top of your feedback.
"""

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    
    return response.text

if __name__ == "__main__":
    # Test Data: A sample physics work energy problem
    sample_question = "A 2 kg block slides down a frictionless ramp from a height of 5 meters. Calculate its final velocity at the bottom. (Take g = 9.8 m/s^2)"
    
    sample_rubric = """
    Total: 5 Points
    - 1 point: Correctly identifying conservation of energy (mgh = 0.5 * m * v^2).
    - 2 points: Correct algebraic isolation of velocity (v = sqrt(2gh)).
    - 2 points: Correct final numerical calculation (~9.9 m/s). 1 point max if units are missing or incorrect.
    """
    
    # Simulating a student who used g=10 instead of 9.8 and dropped their units
    sample_student_work = """
    Initial Energy = mgh = (2)(10)(5) = 100 J.
    Final Energy = 0.5 * m * v^2 = 0.5 * 2 * v^2 = v^2.
    100 = v^2
    v = 10
    """
    
    print("--- Running GradeScribe prototype ---")
    feedback = grade_response(
        student_response=sample_student_work,
        rubric=sample_rubric,
        question=sample_question
    )
    
    print("\n=== GEMINI GRADING BREAKDOWN ===")
    print(feedback)
    print("=================================")