import React from "react";
import { useSelector } from "react-redux";
import {
  Avatar,
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  styled,
} from "@mui/material";

const StyledTableCell = styled(TableCell)(({ theme }) => ({
  "&.MuiTableCell-head": {
    backgroundColor: theme.palette.common.black,
    color: theme.palette.common.white,
  },
  "&.MuiTableCell-body": {
    fontSize: 14,
  },
}));

const Leaderboard = () => {
  
  const users = useSelector((state) => state.user.users);

  return (
    <Box p={4}>
      <Typography variant="h4" my={2}>
        Leaderboard
      </Typography>
      <TableContainer component={Paper}>
        <Table sx={{ minWidth: 700 }}>
          <TableHead>
            <TableRow>
              <StyledTableCell>Users</StyledTableCell>
              <StyledTableCell align="center">Answered</StyledTableCell>
              <StyledTableCell align="center">Create</StyledTableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {users?.map((user, idx) => (
              <TableRow key={idx}>
                <TableCell sx={{ display: "flex", alignItems: "center" }}>
                  <Avatar src={user?.avatarURL} sx={{ mr: 1 }} /> {user?.name} -{" "}
                  {user?.id}
                </TableCell>
                <TableCell align="center">
                  {Object.values(user.answers).length}
                </TableCell>
                <TableCell align="center">{user.questions.length}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default Leaderboard;
