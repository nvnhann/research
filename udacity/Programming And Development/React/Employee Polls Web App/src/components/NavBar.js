import * as React from 'react';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import Button from '@mui/material/Button';
import AdbIcon from '@mui/icons-material/Adb';
import { Link, useNavigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { logout } from '../store/userSlice';

const pages = [
    { name: "Home", url: "/"},
    { name: "Leaderboard", url: "/leaderboard"},
    { name: "New Poll", url: "/new"},
];

function NavBar() {

    const navigate = useNavigate();
    const user = useSelector(state => state.user?.user);
    const dispatch = useDispatch();
    const isLoggin = !!user?.id;

  return (
    <AppBar position="static">
      <Container maxWidth="xl">
        <Toolbar disableGutters>
          <AdbIcon sx={{ display: { xs: 'none', md: 'flex' }, mr: 1 }} />
          <Typography
            variant="h6"
            noWrap
            component={Link}
            to="/"
            sx={{
              mr: 2,
              display: { xs: 'none', md: 'flex' },
              fontFamily: 'monospace',
              fontWeight: 700,
              letterSpacing: '.3rem',
              color: 'inherit',
              textDecoration: 'none',
            }}
          >
            NhanNV13
          </Typography>

          <Box sx={{ flexGrow: 1, display: { xs: 'none', md: 'flex' } }}>
            {pages.map((page, idx) => (
              <Button
                key={idx}
                onClick={() => navigate(page.url)}
                sx={{ my: 2, color: 'white', display: 'block' }}
              >
                {page.name}
              </Button>
            ))}
          </Box>
          {isLoggin && <Typography mx={2}>Hi! {user?.name}</Typography>}
          <Box sx={{ flexGrow: 0 }}>

            {isLoggin && <Button onClick={() => dispatch(logout())} variant='contained' sx={{ color: '#fff', backgroundColor: '#8863FF'}}>
                Logout
            </Button>}
            {!isLoggin && <Button onClick={()=>navigate('/login')} variant='contained' sx={{ color: '#fff', backgroundColor: '#8863FF'}}>
                Login
            </Button>}
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
}
export default NavBar;