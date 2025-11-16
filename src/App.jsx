import { useState, useEffect } from 'react';
import { BookOpen, CheckCircle, XCircle, Flag, ChevronLeft, ChevronRight, Home, RotateCcw, Lightbulb } from 'lucide-react';
import chaptersData from './data/chapters';
import { storage } from './utils/storage';

function App() {
  const [studyMode, setStudyMode] = useState(null); // 'practice', 'test', 'review'
  const [currentChapter, setCurrentChapter] = useState(null);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [userAnswers, setUserAnswers] = useState({});
  const [showHint, setShowHint] = useState(false);
  const [testSubmitted, setTestSubmitted] = useState(false);
  const [flaggedQuestions, setFlaggedQuestions] = useState([]);

  useEffect(() => {
    if (currentChapter) {
      const answers = storage.getUserAnswers(currentChapter);
      setUserAnswers(answers);
      const flagged = storage.getFlaggedQuestions(currentChapter);
      setFlaggedQuestions(flagged);
    }
  }, [currentChapter]);

  const handleModeSelect = (mode) => {
    setStudyMode(mode);
    setCurrentChapter(null);
  };

  const handleChapterSelect = (chapterId) => {
    setCurrentChapter(chapterId);
    setCurrentQuestionIndex(0);
    setShowHint(false);
    setTestSubmitted(false);
  };

  const handleAnswer = (questionId, answerIndex) => {
    const newAnswers = { ...userAnswers, [questionId]: answerIndex };
    setUserAnswers(newAnswers);
    storage.saveUserAnswer(currentChapter, questionId, answerIndex);
    setShowHint(false);
  };

  const handleToggleFlag = (questionId) => {
    storage.toggleFlaggedQuestion(currentChapter, questionId);
    const flagged = storage.getFlaggedQuestions(currentChapter);
    setFlaggedQuestions(flagged);
  };

  const handleResetChapter = () => {
    if (window.confirm('Are you sure you want to reset all progress for this chapter?')) {
      storage.resetChapter(currentChapter);
      setUserAnswers({});
      setFlaggedQuestions([]);
      setCurrentQuestionIndex(0);
    }
  };

  const handleBack = () => {
    if (currentChapter) {
      setCurrentChapter(null);
      setTestSubmitted(false);
    } else {
      setStudyMode(null);
    }
  };

  const getChapterStats = (chapterId) => {
    const answers = storage.getUserAnswers(chapterId);
    const chapter = chaptersData[chapterId];
    const totalQuestions = chapter.questions.length;
    const answeredCount = Object.keys(answers).length;

    let correctCount = 0;
    chapter.questions.forEach(q => {
      if (answers[q.id] !== undefined && answers[q.id] === q.correct) {
        correctCount++;
      }
    });

    return {
      total: totalQuestions,
      answered: answeredCount,
      correct: correctCount,
      accuracy: answeredCount > 0 ? Math.round((correctCount / answeredCount) * 100) : 0
    };
  };

  const renderDashboard = () => {
    const chapters = Object.entries(chaptersData);

    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-800 mb-2">Quantitative Finance LMS</h1>
            <p className="text-gray-600">Master quantitative finance concepts through practice</p>
          </div>

          {!studyMode ? (
            <div className="grid md:grid-cols-3 gap-6 mb-8">
              <button
                onClick={() => handleModeSelect('practice')}
                className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow"
              >
                <div className="text-blue-600 mb-4">
                  <BookOpen size={48} className="mx-auto" />
                </div>
                <h3 className="text-xl font-semibold mb-2">Practice Mode</h3>
                <p className="text-gray-600">Get instant feedback on each question</p>
              </button>

              <button
                onClick={() => handleModeSelect('test')}
                className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow"
              >
                <div className="text-green-600 mb-4">
                  <CheckCircle size={48} className="mx-auto" />
                </div>
                <h3 className="text-xl font-semibold mb-2">Test Mode</h3>
                <p className="text-gray-600">Answer all questions before seeing results</p>
              </button>

              <button
                onClick={() => handleModeSelect('review')}
                className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow"
              >
                <div className="text-orange-600 mb-4">
                  <Flag size={48} className="mx-auto" />
                </div>
                <h3 className="text-xl font-semibold mb-2">Review Mode</h3>
                <p className="text-gray-600">Focus on flagged and incorrect questions</p>
              </button>
            </div>
          ) : (
            <div>
              <button
                onClick={handleBack}
                className="mb-6 flex items-center text-blue-600 hover:text-blue-800"
              >
                <ChevronLeft size={20} />
                Back to Mode Selection
              </button>

              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {chapters.map(([chapterId, chapter]) => {
                  const stats = getChapterStats(chapterId);

                  return (
                    <div key={chapterId} className="bg-white rounded-lg shadow-md p-6">
                      <h3 className="text-xl font-semibold mb-2 text-gray-800">{chapter.title}</h3>
                      <p className="text-gray-600 mb-4 text-sm">{chapter.description}</p>

                      <div className="mb-4 space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">Progress</span>
                          <span className="font-semibold">{stats.answered}/{stats.total}</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full"
                            style={{ width: `${(stats.answered / stats.total) * 100}%` }}
                          />
                        </div>
                        {stats.answered > 0 && (
                          <div className="text-sm text-gray-600">
                            Accuracy: <span className="font-semibold text-green-600">{stats.accuracy}%</span>
                          </div>
                        )}
                      </div>

                      <button
                        onClick={() => handleChapterSelect(chapterId)}
                        className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition-colors"
                      >
                        Start {studyMode === 'practice' ? 'Practice' : studyMode === 'test' ? 'Test' : 'Review'}
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderQuestion = () => {
    const chapter = chaptersData[currentChapter];
    const question = chapter.questions[currentQuestionIndex];
    const userAnswer = userAnswers[question.id];
    const isAnswered = userAnswer !== undefined;
    const isCorrect = isAnswered && userAnswer === question.correct;
    const isFlagged = flaggedQuestions.includes(question.id);
    const showFeedback = studyMode === 'practice' ? isAnswered : testSubmitted;

    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
        <div className="max-w-4xl mx-auto">
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <div className="flex justify-between items-center mb-6">
              <button
                onClick={handleBack}
                className="flex items-center text-blue-600 hover:text-blue-800"
              >
                <Home size={20} className="mr-1" />
                Dashboard
              </button>
              <button
                onClick={handleResetChapter}
                className="flex items-center text-red-600 hover:text-red-800"
              >
                <RotateCcw size={20} className="mr-1" />
                Reset Chapter
              </button>
            </div>

            <div className="mb-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-2">{chapter.title}</h2>
              <div className="flex justify-between items-center text-sm text-gray-600">
                <span>Question {currentQuestionIndex + 1} of {chapter.questions.length}</span>
                <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full">
                  {question.difficulty}
                </span>
              </div>
            </div>

            <div className="mb-6">
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-lg font-semibold text-gray-800">{question.question}</h3>
                <button
                  onClick={() => handleToggleFlag(question.id)}
                  className={`ml-4 ${isFlagged ? 'text-orange-500' : 'text-gray-400'} hover:text-orange-600`}
                >
                  <Flag size={24} fill={isFlagged ? 'currentColor' : 'none'} />
                </button>
              </div>

              <div className="space-y-3">
                {question.options.map((option, index) => {
                  const isSelected = userAnswer === index;
                  const isCorrectOption = index === question.correct;

                  let buttonClass = 'w-full text-left p-4 rounded-lg border-2 transition-all ';

                  if (showFeedback) {
                    if (isCorrectOption) {
                      buttonClass += 'border-green-500 bg-green-50 ';
                    } else if (isSelected && !isCorrect) {
                      buttonClass += 'border-red-500 bg-red-50 ';
                    } else {
                      buttonClass += 'border-gray-300 ';
                    }
                  } else {
                    buttonClass += isSelected
                      ? 'border-blue-500 bg-blue-50 '
                      : 'border-gray-300 hover:border-blue-300 ';
                  }

                  return (
                    <button
                      key={index}
                      onClick={() => !showFeedback && handleAnswer(question.id, index)}
                      disabled={showFeedback}
                      className={buttonClass}
                    >
                      <div className="flex items-center">
                        <span className="font-semibold mr-3">{String.fromCharCode(65 + index)}.</span>
                        <span>{option}</span>
                        {showFeedback && isCorrectOption && (
                          <CheckCircle className="ml-auto text-green-600" size={20} />
                        )}
                        {showFeedback && isSelected && !isCorrect && (
                          <XCircle className="ml-auto text-red-600" size={20} />
                        )}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {showFeedback && (
              <div className={`p-4 rounded-lg mb-6 ${isCorrect ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}`}>
                <p className="font-semibold mb-2 flex items-center">
                  {isCorrect ? (
                    <>
                      <CheckCircle className="mr-2 text-green-600" size={20} />
                      Correct!
                    </>
                  ) : (
                    <>
                      <XCircle className="mr-2 text-red-600" size={20} />
                      Incorrect
                    </>
                  )}
                </p>
                <p className="text-gray-700">{question.explanation}</p>
              </div>
            )}

            {!showFeedback && (
              <div className="flex gap-4">
                <button
                  onClick={() => setShowHint(!showHint)}
                  className="flex items-center text-blue-600 hover:text-blue-800"
                >
                  <Lightbulb size={20} className="mr-1" />
                  {showHint ? 'Hide' : 'Show'} Hint
                </button>
              </div>
            )}

            {showHint && !showFeedback && (
              <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                <p className="text-gray-700">{question.hint}</p>
              </div>
            )}

            <div className="flex justify-between mt-6">
              <button
                onClick={() => setCurrentQuestionIndex(Math.max(0, currentQuestionIndex - 1))}
                disabled={currentQuestionIndex === 0}
                className="flex items-center px-4 py-2 bg-gray-200 rounded-lg hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft size={20} />
                Previous
              </button>

              {studyMode === 'test' && currentQuestionIndex === chapter.questions.length - 1 && !testSubmitted && (
                <button
                  onClick={() => setTestSubmitted(true)}
                  className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                >
                  Submit Test
                </button>
              )}

              <button
                onClick={() => setCurrentQuestionIndex(Math.min(chapter.questions.length - 1, currentQuestionIndex + 1))}
                disabled={currentQuestionIndex === chapter.questions.length - 1}
                className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
                <ChevronRight size={20} />
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  if (currentChapter) {
    return renderQuestion();
  }

  return renderDashboard();
}

export default App;
