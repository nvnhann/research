import { Divider, Grid, Typography } from "@mui/material";
import { useSelector } from "react-redux";
import Card from "../components/Card";

export default function DashBoard() {
  const questions = useSelector((state) => state.question.questions);
  const user = useSelector((state) => state.user.user);
  const users = useSelector((state) => state.user.users);

  return (
    <>
      <Typography variant="h4" my={2}>
        DashBoard
      </Typography>
      <Divider />
      <Typography variant="h5" my={2}>
        New Question
      </Typography>
      <Grid container spacing={1}>
        {questions
          ?.filter(
            (question) =>
              (
                !question.optionOne.votes.includes(user.id) &&
                
                !question.optionTwo.votes.includes(user.id)
              )
          )
          .map((question) => (
            <Grid key={question.id} item xs={6} md={3} lg={2}>
              <Card
                question={question}
                author={users.find((user) => user.id === question.author)}
              />
            </Grid>
          ))}
      </Grid>
      <Divider sx={{ margin: "1rem 0" }} />
      <Typography variant="h5" my={2}>
        Answered Questions
      </Typography>
      <Grid container spacing={1}>
        {questions
          ?.filter(
            (question) =>
              question.optionOne.votes.includes(user.id) ||
              question.optionTwo.votes.includes(user.id)
          )
          .map((question) => (
            <Grid key={question.id} item xs={6} md={3} lg={2}>
              <Card
                question={question}
                author={users.find((user) => user.id === question.author)}
              />
            </Grid>
          ))}
      </Grid>
    </>
  );
}
