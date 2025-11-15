// CHAPTER TEMPLATE
// Copy this template to add a new chapter to src/data/chapters.js

'YOUR-CHAPTER-ID': {
  title: 'Your Chapter Title',
  description: 'Brief description of what this chapter covers',
  questions: [
    {
      id: 1,
      question: "Your question here?",
      options: [
        "Option A",
        "Option B", 
        "Option C",
        "Option D"
      ],
      correct: 0,  // Change to index of correct answer (0-3)
      explanation: "Detailed explanation of the correct answer and why other options are incorrect.",
      difficulty: "Basic",  // Basic, Intermediate, or Advanced
      concept: "Concept Name",
      hint: "Helpful hint without revealing the answer"
    },
    {
      id: 2,
      question: "Second question?",
      options: [
        "Option A",
        "Option B",
        "Option C", 
        "Option D"
      ],
      correct: 1,
      explanation: "Explanation for question 2.",
      difficulty: "Intermediate",
      concept: "Another Concept",
      hint: "Hint for question 2"
    },
    // Add more questions following the same pattern
    // Remember to increment the id for each question
    // Ensure each question has all required fields
  ]
}

// INSTRUCTIONS:
// 1. Replace 'YOUR-CHAPTER-ID' with a unique ID (e.g., 'QF-02', 'ML-03')
// 2. Update title and description
// 3. Add your questions following the template above
// 4. Ensure 'correct' index matches your intended answer (0 = first option)
// 5. Save and test in the application
