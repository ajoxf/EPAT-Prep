# Adding New Chapters Guide

This guide will help you add new chapters to the Quantitative Finance LMS.

## Quick Start

1. Open `src/data/chapters.js`
2. Copy the existing chapter structure
3. Modify the chapter ID, title, description, and questions
4. Save the file
5. The new chapter will automatically appear on the dashboard!

## Step-by-Step Guide

### Step 1: Open the Chapters File

Navigate to `src/data/chapters.js` in your project.

### Step 2: Add Your Chapter

Add a new entry to the `chaptersData` object:

```javascript
const chaptersData = {
  'MLT-01': {
    // ... existing chapter ...
  },
  
  // ADD YOUR NEW CHAPTER HERE
  'CHAPTER-ID': {
    title: 'Your Chapter Title',
    description: 'Brief description of what this chapter covers',
    questions: [
      // Questions go here
    ]
  }
};
```

### Step 3: Chapter ID Naming Convention

Use a consistent naming pattern:
- Format: `SUBJECT-NUMBER`
- Examples:
  - `MLT-01` (Machine Learning Topics - 1)
  - `MLT-02` (Machine Learning Topics - 2)
  - `QF-01` (Quantitative Finance - 1)
  - `STATS-01` (Statistics - 1)
  - `CALC-01` (Calculus - 1)

### Step 4: Add Questions

Each question must follow this exact structure:

```javascript
{
  id: 1,  // Unique ID within the chapter (start from 1)
  
  question: "What is the question text?",
  
  options: [
    "First option",
    "Second option",
    "Third option",
    "Fourth option"
  ],
  
  correct: 1,  // Index of correct answer (0 = first, 1 = second, etc.)
  
  explanation: "Detailed explanation of why this is the correct answer and why other options are wrong.",
  
  difficulty: "Intermediate",  // Choose: "Basic", "Intermediate", or "Advanced"
  
  concept: "Main Concept Name",  // e.g., "Neural Networks", "Regression", etc.
  
  hint: "A helpful hint without giving away the answer."
}
```

### Step 5: Complete Example

Here's a complete example of a new chapter:

```javascript
'QF-01': {
  title: 'Derivatives Basics',
  description: 'Introduction to derivatives, options, futures, and basic pricing',
  questions: [
    {
      id: 1,
      question: "What is a call option?",
      options: [
        "An obligation to buy an asset",
        "A right to buy an asset at a specified price",
        "An obligation to sell an asset",
        "A right to sell an asset at a specified price"
      ],
      correct: 1,
      explanation: "A call option gives the holder the RIGHT (not obligation) to BUY an asset at a specified strike price before expiration. Put options give the right to sell.",
      difficulty: "Basic",
      concept: "Options",
      hint: "Think about what happens when you 'call' something - you bring it to you (buy it)."
    },
    {
      id: 2,
      question: "In the Black-Scholes model, what does volatility represent?",
      options: [
        "The average price of the stock",
        "The standard deviation of returns",
        "The risk-free rate",
        "The strike price"
      ],
      correct: 1,
      explanation: "Volatility (σ) in Black-Scholes represents the standard deviation of the stock's returns, measuring price uncertainty. Higher volatility increases option value due to greater potential price movements.",
      difficulty: "Intermediate",
      concept: "Black-Scholes Model",
      hint: "Volatility measures how much prices fluctuate - what statistical measure describes variation?"
    },
    // Add more questions...
  ]
}
```

## Best Practices

### Question Writing Tips

1. **Clear and Concise**: Make questions unambiguous
2. **Realistic Options**: All options should be plausible
3. **Detailed Explanations**: Help students learn, don't just say "correct"
4. **Good Hints**: Guide thinking without revealing the answer
5. **Proper Difficulty**: Match difficulty to actual question complexity

### Difficulty Guidelines

- **Basic**: Definitions, simple concepts, recall
- **Intermediate**: Application, analysis, multi-step thinking
- **Advanced**: Complex scenarios, synthesis, expert-level

### Concept Tagging

Use consistent concept names across questions:
- "Options Pricing"
- "Risk Management"
- "Portfolio Theory"
- "Time Series Analysis"

### Common Mistakes to Avoid

❌ Don't forget commas between questions
❌ Don't use quotes inside strings without escaping
❌ Don't make all correct answers the same index
❌ Don't leave any fields empty
❌ Don't use special characters that need escaping

✅ Do validate JSON structure
✅ Do test questions in Practice mode
✅ Do provide detailed explanations
✅ Do randomize correct answer positions

## Testing Your Chapter

After adding a new chapter:

1. Save the file
2. Refresh the application (if running)
3. Check that the chapter appears on the dashboard
4. Click into the chapter and test several questions
5. Verify:
   - All options display correctly
   - Correct answer is highlighted properly
   - Explanations are clear
   - Hints are helpful
   - Difficulty badges show correctly

## Chapter Organization Tips

### Structuring Questions

Organize questions by:
1. **Topic progression**: Start simple, build complexity
2. **Concept grouping**: Related concepts together
3. **Difficulty curve**: Mix difficulties throughout
4. **Practical application**: Include real-world scenarios

### Example Chapter Structure

```
Questions 1-10:   Basic definitions and concepts
Questions 11-20:  Intermediate applications
Questions 21-30:  Advanced problem-solving
Questions 31-40:  Mixed review and real-world
```

## Advanced: Multiple Topics

If you have many chapters, organize them by subject:

```javascript
const chaptersData = {
  // Machine Learning
  'ML-01': { ... },
  'ML-02': { ... },
  
  // Quantitative Finance
  'QF-01': { ... },
  'QF-02': { ... },
  
  // Statistics
  'STAT-01': { ... },
  'STAT-02': { ... },
};
```

## Troubleshooting

### Chapter Doesn't Appear

- Check for syntax errors (missing commas, brackets)
- Ensure chapter ID is unique
- Verify the file saved properly
- Refresh the browser

### Questions Display Incorrectly

- Check that `correct` index matches an option (0-3)
- Verify all string quotes are properly closed
- Ensure arrays have 4 options
- Look for special characters needing escaping

### Progress Not Saving

- Storage uses chapter ID + question ID
- Changing IDs will reset progress
- Keep IDs consistent across updates

## Quick Reference

### Minimal Question Template

```javascript
{
  id: 1,
  question: "Question text?",
  options: ["A", "B", "C", "D"],
  correct: 0,
  explanation: "Explanation text.",
  difficulty: "Basic",
  concept: "Concept Name",
  hint: "Hint text."
}
```

### Full Chapter Template

```javascript
'TOPIC-##': {
  title: 'Chapter Title',
  description: 'What students will learn',
  questions: [
    { /* question 1 */ },
    { /* question 2 */ },
    // ... more questions
  ]
}
```

## Need Help?

- Check existing chapters for examples
- Validate JSON syntax using an online validator
- Test in Practice mode before sharing
- Open an issue on GitHub for questions

Happy teaching! 🎓
