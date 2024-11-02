import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { _getUsers } from "../utils/_DATA";

export const Login = createAsyncThunk("user/login", async (payload) => {
    const { username, password } = payload;
    const response = await _getUsers();
    const user = Object.values(response).find(
      (user) => user.id === username && user.password === password
    );
    if (!user) {
      throw new Error("Invalid username or password");
    }
    return user;
});

export const fetchUsers = createAsyncThunk("user/fetchUsers", async () => {
    const response = await _getUsers();
    return Object.values(response).sort((a, b) => Object.keys(b.answers).length - Object.keys(a.answers).length)
});

const userSlice = createSlice({
  name: "user",
  initialState: {
   user: {},
   users: []
  },
  reducers: {
    logout(state) {
      state.user = {};
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(Login.fulfilled, (state, action) => {
        state.user = action.payload;
      })
      .addCase(fetchUsers.fulfilled, (state, action) => {
        state.users = action.payload;
      })
  },
});
const { actions, reducer } = userSlice;
export const { logout } = actions;
export default reducer;