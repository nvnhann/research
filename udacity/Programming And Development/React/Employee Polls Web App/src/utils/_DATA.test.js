const {
    _getUsers,
    _getQuestions,
    _saveQuestion,
    _saveQuestionAnswer
  } = require('./_DATA');
  
  describe('API Tests', () => {
    describe('_getUsers', () => {
      it('should return the users object', async () => {
        const users = await _getUsers();
        expect(users).toBeDefined();
        expect(typeof users).toBe('object');
      });
    });
  
    describe('_getQuestions', () => {
      it('should return the questions object', async () => {
        const questions = await _getQuestions();
        expect(questions).toBeDefined();
        expect(typeof questions).toBe('object');
      });
    });
  
    describe('_saveQuestion', () => {
      it('should add a new question to the questions object', async () => {
        const newQuestion = {
          optionOneText: 'Test Option One',
          optionTwoText: 'Test Option Two',
          author: 'sarahedo'
        };
        const savedQuestion = await _saveQuestion(newQuestion);
        const questions = await _getQuestions();
  
        expect(savedQuestion).toBeDefined();
        expect(typeof savedQuestion).toBe('object');
        expect(Object.values(questions).includes(savedQuestion)).toBeTruthy();
      });
    });
  
    describe('_saveQuestionAnswer', () => {
      it('should add the user answer to the selected question', async () => {
        const authedUser = 'sarahedo';
        const qid = '8xf0y6ziyjabvozdd253nd';
        const answer = 'optionTwo';
  
        await _saveQuestionAnswer({ authedUser, qid, answer });
        const users = await _getUsers();
        const questions = await _getQuestions();
  
        expect(users[authedUser].answers[qid]).toBe(answer);
        expect(questions[qid][answer].votes.includes(authedUser)).toBeTruthy();
      });
    });
  });
  