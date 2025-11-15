# Quantitative Finance Learning Management System (LMS)

A modern, interactive learning platform for quantitative finance exam preparation with multiple study modes, progress tracking, and comprehensive question banks.

## Features

- **📚 Multiple Study Modes**
  - Practice Mode: Instant feedback on each question
  - Test Mode: Answer all questions before seeing results
  - Review Mode: Focus on flagged and incorrect questions

- **📊 Progress Tracking**
  - Track answered questions, correct answers, and accuracy
  - Visual progress indicators for each chapter
  - Persistent storage using localStorage

- **🎯 Smart Learning**
  - Difficulty levels (Basic, Intermediate, Advanced)
  - Concept tagging for organized learning
  - Hints and detailed explanations
  - Question flagging for later review

- **🎨 Modern UI**
  - Clean, responsive design
  - Gradient backgrounds
  - Smooth transitions
  - Mobile-friendly interface

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Lucide React** - Icons
- **localStorage** - Progress persistence

## Getting Started

### Prerequisites

- Node.js 16+ and npm/yarn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quant-finance-lms.git
cd quant-finance-lms
```

2. Install dependencies:
```bash
npm install
```

3. Install Tailwind CSS and its dependencies:
```bash
npm install -D tailwindcss postcss autoprefixer
```

4. Start the development server:
```bash
npm run dev
```

5. Open your browser and navigate to `http://localhost:5173`

### Build for Production

```bash
npm run build
```

The built files will be in the `dist` folder, ready to deploy.

## Project Structure

```
quant-finance-lms/
├── src/
│   ├── data/
│   │   └── chapters.js          # Chapter and question data
│   ├── utils/
│   │   └── storage.js           # localStorage wrapper
│   ├── App.jsx                  # Main application component
│   ├── main.jsx                 # React entry point
│   └── index.css                # Global styles with Tailwind
├── public/                      # Static assets
├── index.html                   # HTML entry point
├── package.json                 # Dependencies and scripts
├── vite.config.js              # Vite configuration
├── tailwind.config.js          # Tailwind configuration
├── postcss.config.js           # PostCSS configuration
└── README.md                    # This file
```

## Adding New Chapters

To add a new chapter, edit `src/data/chapters.js`:

```javascript
const chaptersData = {
  // ... existing chapters ...
  
  'MLT-02': {
    title: 'Machine Learning-II',
    description: 'Neural Networks and Deep Learning',
    questions: [
      {
        id: 1,
        question: "What is a neural network?",
        options: [
          "Option A",
          "Option B",
          "Option C",
          "Option D"
        ],
        correct: 1,  // Index of correct answer (0-based)
        explanation: "Detailed explanation here...",
        difficulty: "Intermediate",  // Basic, Intermediate, or Advanced
        concept: "Neural Networks",
        hint: "Think about biological neurons..."
      },
      // Add more questions...
    ]
  }
};
```

### Question Object Structure

- `id`: Unique identifier within the chapter (number)
- `question`: The question text (string)
- `options`: Array of 4 answer options (array of strings)
- `correct`: Index of the correct answer, 0-based (number: 0, 1, 2, or 3)
- `explanation`: Detailed explanation of the answer (string)
- `difficulty`: Difficulty level (string: "Basic", "Intermediate", or "Advanced")
- `concept`: Main concept being tested (string)
- `hint`: Helpful hint for the student (string)

## Usage

### For Students

1. **Select Study Mode**: Choose between Practice, Test, or Review mode
2. **Choose a Chapter**: Click "Start Practice" on any chapter
3. **Answer Questions**: 
   - Click on an answer option to select it
   - In Practice mode, you'll see instant feedback
   - In Test mode, review all answers at the end
4. **Use Learning Aids**:
   - Click "Show Hint" for guidance
   - Flag questions for later review
   - Read explanations after answering
5. **Track Progress**: View your stats on the dashboard

### For Instructors/Content Creators

1. Add chapters by editing `src/data/chapters.js`
2. Follow the question structure guidelines
3. Ensure each question has all required fields
4. Test questions in Practice mode before sharing
5. Use difficulty levels and concept tags consistently

## Features in Detail

### Study Modes

- **Practice Mode**: Immediate feedback after each answer, perfect for learning
- **Test Mode**: Simulate exam conditions, see results after completing all questions
- **Review Mode**: Focus on previously flagged or incorrectly answered questions

### Progress Tracking

- Automatically saves progress to browser localStorage
- Tracks which questions you've attempted
- Records correct/incorrect answers
- Maintains flagged questions list
- Calculates accuracy percentage per chapter

### Visual Indicators

- **Green**: Correct answers
- **Red**: Incorrect answers
- **Blue**: Selected but not yet graded
- **Orange**: Flagged questions
- **Progress bars**: Show completion percentage

## Deployment

### Deploy to GitHub Pages

1. Update `vite.config.js`:
```javascript
export default defineConfig({
  plugins: [react()],
  base: '/quant-finance-lms/', // Your repo name
})
```

2. Build and deploy:
```bash
npm run build
# Deploy the dist folder to GitHub Pages
```

### Deploy to Netlify/Vercel

1. Connect your GitHub repository
2. Set build command: `npm run build`
3. Set publish directory: `dist`
4. Deploy!

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

MIT License - feel free to use this project for your own learning platform!

## Support

For questions or issues, please open an issue on GitHub.

## Roadmap

- [ ] Analytics dashboard with charts
- [ ] Export progress reports
- [ ] Timed quiz mode
- [ ] Collaborative study features
- [ ] Mobile app version
- [ ] More chapters and topics
- [ ] Custom quiz creation tool
- [ ] Spaced repetition algorithm

## Acknowledgments

- Built with React and Vite
- Icons by Lucide
- Styled with Tailwind CSS

---

Happy Learning! 🎓
