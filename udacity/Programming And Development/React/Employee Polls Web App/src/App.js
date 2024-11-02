import "./App.css";
import { Route, Routes } from "react-router-dom";
import LoginPage from "./pages/LoginPage";
import { useDispatch } from "react-redux";
import { useEffect } from "react";
import NavBar from "./components/NavBar";
import { fetchUsers } from "./store/userSlice";
import Leaderboard from "./pages/LeaderBoard";
import { fetchQuestions } from "./store/questionSlice";
import HomePage from "./pages/DashBoard";
import PollPage from "./pages/PollPage";
import PrivateRoute from "./components/PrivateRoute";
import NewPollPage from "./pages/NewPollPage";

function App() {
  const dispatch = useDispatch();
  useEffect(() => {
    (async () => {
      dispatch(await fetchUsers());
      dispatch(await fetchQuestions());
    })();
  }, [dispatch]);

  return (
    <div className="App">
      <NavBar />
      <Routes>
        <Route
          path="/"
          element={
            <PrivateRoute>
              <HomePage />
            </PrivateRoute>
          }
        />
        <Route
          path="/questions/:id"
          element={
            <PrivateRoute>
              <PollPage />
            </PrivateRoute>
          }
        />
        <Route path="/login" exact element={<LoginPage />} />
        <Route
          path="/leaderboard"
          exact
          element={
            <PrivateRoute>
              <Leaderboard />
            </PrivateRoute>
          }
        />
        <Route
          path="/new"
          exac
          element={
            <PrivateRoute>
              <NewPollPage />
            </PrivateRoute>
          }
        />
      </Routes>
    </div>
  );
}

export default App;
