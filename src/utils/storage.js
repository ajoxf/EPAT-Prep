// localStorage wrapper for managing user progress

const STORAGE_KEYS = {
  USER_ANSWERS: 'user-answers',
  FLAGGED_QUESTIONS: 'flagged-questions'
};

export const storage = {
  // Get user answers for a specific chapter
  getUserAnswers(chapterId) {
    try {
      const data = localStorage.getItem(STORAGE_KEYS.USER_ANSWERS);
      const allAnswers = data ? JSON.parse(data) : {};
      return allAnswers[chapterId] || {};
    } catch (error) {
      console.error('Error reading user answers:', error);
      return {};
    }
  },

  // Save user answer for a specific question
  saveUserAnswer(chapterId, questionId, answer) {
    try {
      const data = localStorage.getItem(STORAGE_KEYS.USER_ANSWERS);
      const allAnswers = data ? JSON.parse(data) : {};

      if (!allAnswers[chapterId]) {
        allAnswers[chapterId] = {};
      }

      allAnswers[chapterId][questionId] = answer;
      localStorage.setItem(STORAGE_KEYS.USER_ANSWERS, JSON.stringify(allAnswers));
    } catch (error) {
      console.error('Error saving user answer:', error);
    }
  },

  // Get flagged questions for a specific chapter
  getFlaggedQuestions(chapterId) {
    try {
      const data = localStorage.getItem(STORAGE_KEYS.FLAGGED_QUESTIONS);
      const allFlagged = data ? JSON.parse(data) : {};
      return allFlagged[chapterId] || [];
    } catch (error) {
      console.error('Error reading flagged questions:', error);
      return [];
    }
  },

  // Toggle flagged status for a question
  toggleFlaggedQuestion(chapterId, questionId) {
    try {
      const data = localStorage.getItem(STORAGE_KEYS.FLAGGED_QUESTIONS);
      const allFlagged = data ? JSON.parse(data) : {};

      if (!allFlagged[chapterId]) {
        allFlagged[chapterId] = [];
      }

      const index = allFlagged[chapterId].indexOf(questionId);
      if (index > -1) {
        allFlagged[chapterId].splice(index, 1);
      } else {
        allFlagged[chapterId].push(questionId);
      }

      localStorage.setItem(STORAGE_KEYS.FLAGGED_QUESTIONS, JSON.stringify(allFlagged));
      return allFlagged[chapterId].includes(questionId);
    } catch (error) {
      console.error('Error toggling flagged question:', error);
      return false;
    }
  },

  // Reset chapter progress
  resetChapter(chapterId) {
    try {
      // Reset answers
      const answersData = localStorage.getItem(STORAGE_KEYS.USER_ANSWERS);
      if (answersData) {
        const allAnswers = JSON.parse(answersData);
        delete allAnswers[chapterId];
        localStorage.setItem(STORAGE_KEYS.USER_ANSWERS, JSON.stringify(allAnswers));
      }

      // Reset flagged questions
      const flaggedData = localStorage.getItem(STORAGE_KEYS.FLAGGED_QUESTIONS);
      if (flaggedData) {
        const allFlagged = JSON.parse(flaggedData);
        delete allFlagged[chapterId];
        localStorage.setItem(STORAGE_KEYS.FLAGGED_QUESTIONS, JSON.stringify(allFlagged));
      }
    } catch (error) {
      console.error('Error resetting chapter:', error);
    }
  },

  // Clear all data
  clearAll() {
    try {
      localStorage.removeItem(STORAGE_KEYS.USER_ANSWERS);
      localStorage.removeItem(STORAGE_KEYS.FLAGGED_QUESTIONS);
    } catch (error) {
      console.error('Error clearing all data:', error);
    }
  }
};
