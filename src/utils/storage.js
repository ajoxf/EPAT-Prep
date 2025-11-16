// localStorage wrapper for progress tracking

const STORAGE_PREFIX = 'quant_lms_';

export const storage = {
  // Get item from localStorage
  get: (key) => {
    try {
      const item = localStorage.getItem(STORAGE_PREFIX + key);
      return item ? JSON.parse(item) : null;
    } catch (error) {
      console.error('Error reading from localStorage:', error);
      return null;
    }
  },

  // Set item in localStorage
  set: (key, value) => {
    try {
      localStorage.setItem(STORAGE_PREFIX + key, JSON.stringify(value));
      return true;
    } catch (error) {
      console.error('Error writing to localStorage:', error);
      return false;
    }
  },

  // Remove item from localStorage
  remove: (key) => {
    try {
      localStorage.removeItem(STORAGE_PREFIX + key);
      return true;
    } catch (error) {
      console.error('Error removing from localStorage:', error);
      return false;
    }
  },

  // Clear all app data from localStorage
  clear: () => {
    try {
      const keys = Object.keys(localStorage);
      keys.forEach(key => {
        if (key.startsWith(STORAGE_PREFIX)) {
          localStorage.removeItem(key);
        }
      });
      return true;
    } catch (error) {
      console.error('Error clearing localStorage:', error);
      return false;
    }
  }
};

// Progress tracking helpers
export const progressStorage = {
  // Get progress for a specific chapter
  getChapterProgress: (chapterId) => {
    return storage.get(`progress_${chapterId}`) || {
      answeredQuestions: {},
      correctAnswers: {},
      flaggedQuestions: new Set()
    };
  },

  // Save progress for a specific chapter
  saveChapterProgress: (chapterId, progress) => {
    // Convert Set to Array for JSON serialization
    const serializable = {
      ...progress,
      flaggedQuestions: Array.from(progress.flaggedQuestions || [])
    };
    return storage.set(`progress_${chapterId}`, serializable);
  },

  // Mark question as answered
  markAnswered: (chapterId, questionId, isCorrect) => {
    const progress = progressStorage.getChapterProgress(chapterId);
    progress.answeredQuestions[questionId] = true;
    progress.correctAnswers[questionId] = isCorrect;
    return progressStorage.saveChapterProgress(chapterId, progress);
  },

  // Toggle flag on question
  toggleFlag: (chapterId, questionId) => {
    const progress = progressStorage.getChapterProgress(chapterId);
    const flags = new Set(progress.flaggedQuestions || []);

    if (flags.has(questionId)) {
      flags.delete(questionId);
    } else {
      flags.add(questionId);
    }

    progress.flaggedQuestions = flags;
    return progressStorage.saveChapterProgress(chapterId, progress);
  },

  // Check if question is flagged
  isFlagged: (chapterId, questionId) => {
    const progress = progressStorage.getChapterProgress(chapterId);
    const flags = new Set(progress.flaggedQuestions || []);
    return flags.has(questionId);
  },

  // Get all progress
  getAllProgress: () => {
    const allProgress = {};
    const keys = Object.keys(localStorage);
    keys.forEach(key => {
      if (key.startsWith(STORAGE_PREFIX + 'progress_')) {
        const chapterId = key.replace(STORAGE_PREFIX + 'progress_', '');
        allProgress[chapterId] = progressStorage.getChapterProgress(chapterId);
      }
    });
    return allProgress;
  },

  // Reset progress for a chapter
  resetChapter: (chapterId) => {
    return storage.remove(`progress_${chapterId}`);
  },

  // Reset all progress
  resetAll: () => {
    return storage.clear();
  }
};
