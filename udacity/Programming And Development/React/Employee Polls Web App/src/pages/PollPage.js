import {
  Avatar,
  Box,
  Button,
  Divider,
  Grid,
  Paper,
  Typography,
} from "@mui/material";
import { useDispatch, useSelector } from "react-redux";
import { useParams } from "react-router-dom";
import { handleAddAnswer } from "../store/questionSlice";
export default function PollPage() {
  const { id } = useParams();
  const questions = useSelector((state) => state.question.questions);
  const users = useSelector((state) => state.user.users);
  const user = useSelector((state) => state.user.user);
  const dispatch = useDispatch();

  const question = questions.find((question) => question.id === id);
  const author = users.find((user) => user.id === question.author);

  const isVoted =
    question.optionOne.votes.includes(user.id) ||
    question.optionTwo.votes.includes(user.id);

  const saveOptionOne = (e) => {
    dispatch(handleAddAnswer({ questionId: question.id, answer: "optionOne" }));
  };
  const saveOptionTwo = (e) => {
    dispatch(handleAddAnswer({ questionId: question.id, answer: "optionTwo" }));
  };

  const calcPotionOne = () => {
    const total =
      question.optionOne.votes.length + question.optionTwo.votes.length;
    return ((question.optionOne.votes.length / total) * 100).toFixed(2) + "%";
  };

  const calcPotionTwo = () => {
    const total =
      question.optionOne.votes.length + question.optionTwo.votes.length;
    return ((question.optionTwo.votes.length / total) * 100).toFixed(2) + "%";
  };

  return (
    <Box>
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        flexDirection="column"
      >
        <Avatar
          alt="Remy Sharp"
          sx={{ width: "10rem", height: "10rem" }}
          src={author.avatarURL}
        />{" "}
        <Typography variant="h5">{author.id}</Typography>
      </Box>
      <Divider />
      <Typography variant="h5">Would you rather?</Typography>
      <Grid container spacing={2}>
        <Grid item lg={6} md={6}>
          <Paper sx={{ padding: 4 }}>
            <Typography>{question.optionOne.text}</Typography>
            <Button
              disabled={isVoted}
              variant="contained"
              fullWidth
              onClick={saveOptionOne}
            >
              {isVoted ? calcPotionOne() : "Click"}
            </Button>
          </Paper>
        </Grid>
        <Grid item lg={6} md={6}>
          <Paper sx={{ padding: 4 }}>
            <Typography>{question.optionTwo.text}</Typography>
            <Button
              disabled={isVoted}
              variant="contained"
              fullWidth
              onClick={saveOptionTwo}
            >
              {isVoted ? calcPotionTwo() : "Click"}
            </Button>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
