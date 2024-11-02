import React from "react";
import Typography from "@mui/material/Typography";

const NotFoundPage = () => {
  return (
    <div>
      <Typography variant="h2" align="center">
        404 Page Not Found
      </Typography>
      <Typography variant="body1" align="center">
        The requested page does not exist.
      </Typography>
    </div>
  );
};

export default NotFoundPage;
