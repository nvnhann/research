import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { _getQuestions, _saveQuestion, _saveQuestionAnswer } from "../utils/_DATA";
import { fetchUsers } from "./userSlice";

export const fetchQuestions = createAsyncThunk("question/fetch", async () => {
  const response = await _getQuestions();
  return Object.values(response).sort((a, b) => b.timestamp - a.timestamp);
});

export const handleAddAnswer = createAsyncThunk(
  "question/addAnswer",
  async ({ questionId, answer }, { getState, dispatch }) => {
    const { user } = getState();
    await _saveQuestionAnswer({ authedUser: user.user.id, qid: questionId, answer });
    dispatch(fetchQuestions()); // Refresh questions after adding answer
  }
);

export const addQuestion = createAsyncThunk(
  "question/addQuestion",
  async ( payload , { getState, dispatch }) => {

    const { user } = getState();
    const { firstOption, secondOption } = payload;
    
    const question = await _saveQuestion({
      optionOneText: firstOption,
      optionTwoText: secondOption,
      author: user.user.id,
    });

    dispatch(fetchUsers());
   
    return question;
  }
);

const questionSlice = createSlice({
  name: "question",
  initialState: {
    questions: [],
  },
  extraReducers: builder => {
    builder.addCase(fetchQuestions.fulfilled, (state, action) => {
        state.questions = action.payload
    })
    .addCase(addQuestion.fulfilled, (state, action) => {
      state.questions.push(action.payload);
    });
  }
});

const { reducer } = questionSlice;
export default reducer;
