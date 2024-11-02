import { Box, Button, TextField, Typography } from "@mui/material";
import { useState } from "react";
import { useDispatch } from "react-redux";
import { addQuestion } from "../store/questionSlice";
import { useNavigate } from "react-router-dom";

export default function NewPollPage(){

    const [optionOne, setOptionOne] = useState("");
    const [optionTwo, setOptionTwo] = useState("");
    
    const dispatch = useDispatch();
    const navigate = useNavigate();

    const  handleSubmit = () =>{
        if(!optionOne || !optionTwo) return;
        dispatch(addQuestion({ optionOne, optionTwo }));
        navigate("/")
    }

    return <>

    <Box p={4}>
        <Typography variant="h4">New Poll</Typography>
        <TextField 
            fullWidth margin="normal" 
            variant="standard" 
            label="First Option" 
            multiline rows={3} 
            value={optionOne} 
            onChange={ e=> setOptionOne(e.target.value)}
        />
        <TextField 
            fullWidth margin="normal" 
            variant="standard" 
            label="Second Option" 
            multiline 
            rows={3} 
            value={optionTwo} 
            onChange={ e=> setOptionTwo(e.target.value)}
        />
        <Button fullWidth variant="outlined" onClick={handleSubmit}>Submit</Button>
    </Box>
    </>
}