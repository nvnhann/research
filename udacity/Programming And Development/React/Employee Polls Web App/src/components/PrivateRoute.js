import { Navigate } from "react-router-dom";
import { useSelector } from "react-redux";
import { useLocation } from "react-router-dom";

const PrivateRoute = ({ children }) => {
  const isLogin = useSelector((state) => !!state.user.user.id);
  const location = useLocation();
  if (!isLogin) {
    return <Navigate to={`/login?redirect=${location.pathname}`} />;
  }

  return children;
};

export default PrivateRoute;
