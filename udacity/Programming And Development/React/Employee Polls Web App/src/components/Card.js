import { Box, Button, Paper, Typography } from "@mui/material";
import PropTypes from "prop-types";
import { useNavigate } from "react-router-dom";

Card.propTypes = {
  question: PropTypes.object.isRequired,
  author: PropTypes.object.isRequired,
};

export default function Card(props) {
  const { question, author } = props;
  const navigate = useNavigate();
  return (
    <>
      <Box component={Paper} sx={{ minWidth: "10rem", padding: 2 }}>
        <Typography>{author?.name}</Typography>
        <Typography variant="caption" color="GrayText">
          {new Date(question.timestamp).toDateString()}
        </Typography>
        <Button
          onClick={() => navigate(`questions/${question?.id}`)}
          variant="outlined"
          fullWidth
        >
          Show
        </Button>
      </Box>
    </>
  );
}
