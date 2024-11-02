import { configureStore } from "@reduxjs/toolkit";
import userReducer from "./userSlice";
import questionReducer from "./questionSlice";

const rootReducer = {
    user: userReducer,
    question: questionReducer
};

const store = configureStore({
    reducer: rootReducer,
});

export default store;