import { useState, useEffect } from 'react';
import { BookOpen, CheckCircle, Flag, Lightbulb, ChevronLeft, ChevronRight, Home, RotateCcw } from 'lucide-react';
import chaptersData from './data/chapters';
import { progressStorage } from './utils/storage';

function App() {
  const [currentView, setCurrentView] = useState('dashboard'); // 'dashboard', 'quiz'
  const [currentChapter, setCurrentChapter] = useState(null);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [showHint, setShowHint] = useState(false);
  const [studyMode, setStudyMode] = useState('practice'); // 'practice', 'test', 'review'
  const [testAnswers, setTestAnswers] = useState({});
  const [testSubmitted, setTestSubmitted] = useState(false);
  const [progress, setProgress] = useState({});

  // Load progress on mount
  useEffect(() => {
    const allProgress = progressStorage.getAllProgress();
    setProgress(allProgress);
  }, []);

  // Get current question
  const getCurrentQuestion = () => {
    if (!currentChapter) return null;
    const chapter = chaptersData[currentChapter];
    return chapter?.questions[currentQuestionIndex];
  };

  // Get chapter progress stats
  const getChapterStats = (chapterId) => {
    const chapter = chaptersData[chapterId];
    const chapterProgress = progress[chapterId] || { answeredQuestions: {}, correctAnswers: {} };

    const totalQuestions = chapter.questions.length;
    const answeredCount = Object.keys(chapterProgress.answeredQuestions).length;
    const correctCount = Object.values(chapterProgress.correctAnswers).filter(Boolean).length;
    const accuracy = answeredCount > 0 ? Math.round((correctCount / answeredCount) * 100) : 0;

    return { totalQuestions, answeredCount, correctCount, accuracy };
  };

  // Start chapter
  const startChapter = (chapterId, mode = 'practice') => {
    setCurrentChapter(chapterId);
    setCurrentQuestionIndex(0);
    setStudyMode(mode);
    setCurrentView('quiz');
    setSelectedAnswer(null);
    setShowExplanation(false);
    setShowHint(false);
    setTestAnswers({});
    setTestSubmitted(false);
  };

  // Handle answer selection
  const handleAnswerSelect = (answerIndex) => {
    if (studyMode === 'practice') {
      setSelectedAnswer(answerIndex);
      setShowExplanation(true);

      const question = getCurrentQuestion();
      const isCorrect = answerIndex === question.correct;

      // Save progress
      progressStorage.markAnswered(currentChapter, question.id, isCorrect);

      // Update local progress state
      const updatedProgress = progressStorage.getAllProgress();
      setProgress(updatedProgress);
    } else if (studyMode === 'test') {
      setTestAnswers({
        ...testAnswers,
        [currentQuestionIndex]: answerIndex
      });
    }
  };

  // Navigate to next question
  const nextQuestion = () => {
    const chapter = chaptersData[currentChapter];
    if (currentQuestionIndex < chapter.questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
      setSelectedAnswer(null);
      setShowExplanation(false);
      setShowHint(false);
    }
  };

  // Navigate to previous question
  const previousQuestion = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
      setSelectedAnswer(null);
      setShowExplanation(false);
      setShowHint(false);
    }
  };

  // Submit test
  const submitTest = () => {
    setTestSubmitted(true);

    // Grade the test and save progress
    const chapter = chaptersData[currentChapter];
    chapter.questions.forEach((question, index) => {
      const userAnswer = testAnswers[index];
      if (userAnswer !== undefined) {
        const isCorrect = userAnswer === question.correct;
        progressStorage.markAnswered(currentChapter, question.id, isCorrect);
      }
    });

    // Update progress
    const updatedProgress = progressStorage.getAllProgress();
    setProgress(updatedProgress);
  };

  // Toggle flag
  const toggleFlag = () => {
    const question = getCurrentQuestion();
    progressStorage.toggleFlag(currentChapter, question.id);
    const updatedProgress = progressStorage.getAllProgress();
    setProgress(updatedProgress);
  };

  // Check if current question is flagged
  const isQuestionFlagged = () => {
    const question = getCurrentQuestion();
    return progressStorage.isFlagged(currentChapter, question.id);
  };

  // Return to dashboard
  const returnToDashboard = () => {
    setCurrentView('dashboard');
    setCurrentChapter(null);
    setCurrentQuestionIndex(0);
    setSelectedAnswer(null);
    setShowExplanation(false);
    setShowHint(false);
    setTestAnswers({});
    setTestSubmitted(false);
  };

  // Reset chapter progress
  const resetChapterProgress = (chapterId) => {
    if (window.confirm('Are you sure you want to reset all progress for this chapter?')) {
      progressStorage.resetChapter(chapterId);
      const updatedProgress = progressStorage.getAllProgress();
      setProgress(updatedProgress);
    }
  };

  // Render Dashboard
  const renderDashboard = () => {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 p-8">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-5xl font-bold text-white mb-4">Quantitative Finance LMS</h1>
            <p className="text-xl text-gray-300">Master your EPAT preparation with interactive learning</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {Object.entries(chaptersData).map(([chapterId, chapter]) => {
              const stats = getChapterStats(chapterId);

              return (
                <div
                  key={chapterId}
                  className="bg-gray-800 rounded-xl shadow-2xl p-6 card-hover border border-gray-700"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center">
                      <BookOpen className="w-8 h-8 text-blue-400 mr-3" />
                      <div>
                        <h2 className="text-xl font-bold text-white">{chapter.title}</h2>
                        <p className="text-sm text-gray-400">{chapterId}</p>
                      </div>
                    </div>
                  </div>

                  <p className="text-gray-300 text-sm mb-4 line-clamp-2">{chapter.description}</p>

                  {/* Progress Stats */}
                  <div className="mb-4">
                    <div className="flex justify-between text-sm text-gray-400 mb-2">
                      <span>Progress</span>
                      <span>{stats.answeredCount}/{stats.totalQuestions} questions</span>
                    </div>
                    <div className="progress-bar">
                      <div
                        className="progress-fill bg-blue-500"
                        style={{ width: `${(stats.answeredCount / stats.totalQuestions) * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="flex justify-between text-sm mb-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-400">{stats.correctCount}</div>
                      <div className="text-gray-400">Correct</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-400">{stats.accuracy}%</div>
                      <div className="text-gray-400">Accuracy</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-400">{stats.totalQuestions}</div>
                      <div className="text-gray-400">Total</div>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="space-y-2">
                    <button
                      onClick={() => startChapter(chapterId, 'practice')}
                      className="w-full btn-primary text-sm"
                    >
                      Practice Mode
                    </button>
                    <button
                      onClick={() => startChapter(chapterId, 'test')}
                      className="w-full btn-secondary text-sm"
                    >
                      Test Mode
                    </button>
                    {stats.answeredCount > 0 && (
                      <button
                        onClick={() => resetChapterProgress(chapterId)}
                        className="w-full px-4 py-2 bg-red-600 text-white rounded-lg text-sm hover:bg-red-700 transition-colors flex items-center justify-center"
                      >
                        <RotateCcw className="w-4 h-4 mr-2" />
                        Reset Progress
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  };

  // Render Quiz View
  const renderQuiz = () => {
    const chapter = chaptersData[currentChapter];
    const question = getCurrentQuestion();

    if (!question) return null;

    const isTestMode = studyMode === 'test';
    const userAnswer = isTestMode ? testAnswers[currentQuestionIndex] : selectedAnswer;

    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 p-8">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="bg-gray-800 rounded-xl shadow-2xl p-6 mb-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <button
                onClick={returnToDashboard}
                className="flex items-center text-gray-300 hover:text-white transition-colors"
              >
                <Home className="w-5 h-5 mr-2" />
                Dashboard
              </button>
              <div className="flex items-center space-x-4">
                <span className="badge badge-{question.difficulty.toLowerCase()}">{question.difficulty}</span>
                <button
                  onClick={toggleFlag}
                  className={`p-2 rounded-lg transition-colors ${
                    isQuestionFlagged() ? 'bg-yellow-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  <Flag className="w-5 h-5" />
                </button>
              </div>
            </div>
            <h2 className="text-2xl font-bold text-white">{chapter.title}</h2>
            <p className="text-gray-400 mt-2">
              Question {currentQuestionIndex + 1} of {chapter.questions.length}
            </p>
            <div className="progress-bar mt-4">
              <div
                className="progress-fill bg-blue-500"
                style={{ width: `${((currentQuestionIndex + 1) / chapter.questions.length) * 100}%` }}
              ></div>
            </div>
          </div>

          {/* Question */}
          <div className="bg-gray-800 rounded-xl shadow-2xl p-8 mb-6 border border-gray-700 fade-in">
            <div className="mb-6">
              <div className="flex items-start justify-between">
                <h3 className="text-2xl font-semibold text-white mb-6">{question.question}</h3>
              </div>

              <div className="space-y-3">
                {question.options.map((option, index) => {
                  let optionClass = 'quiz-option';

                  if (isTestMode && testSubmitted) {
                    if (index === question.correct) {
                      optionClass += ' correct';
                    } else if (index === userAnswer && index !== question.correct) {
                      optionClass += ' incorrect';
                    }
                  } else if (!isTestMode && showExplanation) {
                    if (index === question.correct) {
                      optionClass += ' correct';
                    } else if (index === selectedAnswer && index !== question.correct) {
                      optionClass += ' incorrect';
                    }
                  } else if (userAnswer === index) {
                    optionClass += ' selected';
                  }

                  const isDisabled = (!isTestMode && showExplanation) || (isTestMode && testSubmitted);
                  if (isDisabled) optionClass += ' disabled';

                  return (
                    <button
                      key={index}
                      onClick={() => !isDisabled && handleAnswerSelect(index)}
                      className={optionClass}
                      disabled={isDisabled}
                    >
                      <div className="flex items-center">
                        <span className="font-semibold mr-3 text-gray-400">
                          {String.fromCharCode(65 + index)}.
                        </span>
                        <span className="text-white text-left">{option}</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Hint */}
            {!isTestMode && (
              <div className="mt-6">
                <button
                  onClick={() => setShowHint(!showHint)}
                  className="flex items-center text-yellow-400 hover:text-yellow-300 transition-colors"
                >
                  <Lightbulb className="w-5 h-5 mr-2" />
                  {showHint ? 'Hide Hint' : 'Show Hint'}
                </button>
                {showHint && (
                  <div className="mt-3 p-4 bg-yellow-900 bg-opacity-30 border border-yellow-600 rounded-lg">
                    <p className="text-yellow-100">{question.hint}</p>
                  </div>
                )}
              </div>
            )}

            {/* Explanation */}
            {((showExplanation && !isTestMode) || (isTestMode && testSubmitted)) && (
              <div className="mt-6 p-6 bg-blue-900 bg-opacity-30 border border-blue-600 rounded-lg fade-in">
                <div className="flex items-center mb-3">
                  <CheckCircle className="w-6 h-6 text-blue-400 mr-2" />
                  <h4 className="text-lg font-semibold text-white">Explanation</h4>
                </div>
                <p className="text-gray-200">{question.explanation}</p>
                <div className="mt-4 pt-4 border-t border-blue-800">
                  <p className="text-sm text-gray-400">
                    <span className="font-semibold">Concept:</span> {question.concept}
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Navigation */}
          <div className="flex justify-between items-center">
            <button
              onClick={previousQuestion}
              disabled={currentQuestionIndex === 0}
              className="flex items-center px-6 py-3 bg-gray-700 text-white rounded-lg font-semibold hover:bg-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronLeft className="w-5 h-5 mr-2" />
              Previous
            </button>

            {isTestMode && !testSubmitted && currentQuestionIndex === chapter.questions.length - 1 && (
              <button
                onClick={submitTest}
                className="btn-success"
              >
                Submit Test
              </button>
            )}

            {currentQuestionIndex < chapter.questions.length - 1 && (
              <button
                onClick={nextQuestion}
                className="flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-colors"
              >
                Next
                <ChevronRight className="w-5 h-5 ml-2" />
              </button>
            )}

            {currentQuestionIndex === chapter.questions.length - 1 && (!isTestMode || testSubmitted) && (
              <button
                onClick={returnToDashboard}
                className="btn-success"
              >
                Finish
              </button>
            )}
          </div>

          {/* Test Results */}
          {isTestMode && testSubmitted && (
            <div className="mt-6 bg-gray-800 rounded-xl shadow-2xl p-6 border border-gray-700">
              <h3 className="text-2xl font-bold text-white mb-4">Test Results</h3>
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center p-4 bg-blue-900 bg-opacity-30 rounded-lg">
                  <div className="text-3xl font-bold text-blue-400">
                    {Object.keys(testAnswers).length}
                  </div>
                  <div className="text-gray-300">Answered</div>
                </div>
                <div className="text-center p-4 bg-green-900 bg-opacity-30 rounded-lg">
                  <div className="text-3xl font-bold text-green-400">
                    {chapter.questions.filter((q, i) => testAnswers[i] === q.correct).length}
                  </div>
                  <div className="text-gray-300">Correct</div>
                </div>
                <div className="text-center p-4 bg-purple-900 bg-opacity-30 rounded-lg">
                  <div className="text-3xl font-bold text-purple-400">
                    {Math.round(
                      (chapter.questions.filter((q, i) => testAnswers[i] === q.correct).length /
                        Object.keys(testAnswers).length) *
                        100
                    )}%
                  </div>
                  <div className="text-gray-300">Score</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return currentView === 'dashboard' ? renderDashboard() : renderQuiz();
}

export default App;
