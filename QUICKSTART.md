# Quick Start Guide

Get your Quantitative Finance LMS up and running in 5 minutes!

## 🚀 Installation (2 minutes)

1. **Clone or download the repository**
   ```bash
   git clone https://github.com/yourusername/quant-finance-lms.git
   cd quant-finance-lms
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```

4. **Open your browser**
   - Navigate to `http://localhost:5173`
   - You should see the dashboard!

That's it! 🎉

## 📚 First Steps (3 minutes)

### Using the Application

1. **Select a Study Mode** on the dashboard:
   - **Practice Mode** - Get instant feedback (recommended for learning)
   - **Test Mode** - Simulate exam conditions
   - **Review Mode** - Focus on flagged questions

2. **Choose a Chapter** and click "Start Practice"

3. **Answer Questions**:
   - Click an option to select it
   - In Practice mode, see immediate feedback
   - Click "Show Hint" if you need help
   - Flag questions for later review

4. **Track Your Progress**:
   - Return to dashboard to see stats
   - View completion percentage
   - Check accuracy rates

### Adding Your First Question

1. **Open** `src/data/chapters.js`

2. **Find** the MLT-01 chapter

3. **Add a question** at the end of the questions array:
   ```javascript
   {
     id: 51,  // Next available ID
     question: "What is your question?",
     options: [
       "Option A",
       "Option B",
       "Option C",
       "Option D"
     ],
     correct: 1,  // Index of correct answer (0-3)
     explanation: "Why this is correct...",
     difficulty: "Basic",
     concept: "Your Concept",
     hint: "Helpful hint here"
   }
   ```

4. **Save** the file and refresh your browser

5. **Test** your new question!

## 🎯 Common Tasks

### Create a New Chapter

1. Open `src/data/chapters.js`
2. Copy an existing chapter structure
3. Change the chapter ID (e.g., 'MLT-02')
4. Update title and description
5. Add your questions
6. Save and refresh!

**Example:**
```javascript
'MLT-02': {
  title: 'Machine Learning II',
  description: 'Advanced ML concepts',
  questions: [
    // Your questions here
  ]
}
```

### Reset Progress

**For a specific chapter:**
- Click "Reset Chapter" button in the chapter view

**For all progress:**
- Open browser DevTools (F12)
- Go to Application/Storage tab
- Find localStorage
- Clear 'user-answers' and 'flagged-questions'

### Build for Production

```bash
npm run build
```

Files will be in the `dist/` folder, ready to deploy!

## 🐛 Troubleshooting

### App won't start?
```bash
# Clear and reinstall
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Questions not saving?
- Check browser console for errors
- Ensure localStorage is enabled
- Try incognito mode to test

### Changes not showing?
- Hard refresh: Ctrl+F5 (Windows) or Cmd+Shift+R (Mac)
- Clear browser cache
- Restart dev server

## 📖 Next Steps

- Read [README.md](README.md) for full documentation
- See [ADDING_CHAPTERS.md](ADDING_CHAPTERS.md) for detailed chapter guide
- Check [DEPLOYMENT.md](DEPLOYMENT.md) to publish your app

## 💡 Tips

- Use Practice Mode while learning
- Flag tricky questions for review
- Check explanations even if you get it right
- Track your progress to identify weak areas
- Add your own questions to reinforce learning

## 🆘 Need Help?

- Check existing questions for examples
- Open an issue on GitHub
- Review the documentation files

## 🎓 Happy Learning!

You're all set! Start practicing and master those quantitative finance concepts.

---

**Remember:**
- Practice Mode = Learning
- Test Mode = Assessment
- Review Mode = Reinforcement
