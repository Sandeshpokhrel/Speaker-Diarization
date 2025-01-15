import React, {useEffect} from "react";
import { useNavigate} from "react-router-dom";
import "./MainPage.css";

const MainPage = () => {
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem("access");
    if (!token) {
      navigate("/login");
    }
  }, [navigate]);


  const handleLogout = () => {
    localStorage.removeItem("access");
    navigate("/login");
  };

  return (
    <div className="second-container">
      {/* Navbar */}
      <div className="navbar">
        <div className="navbar-left">
          <button onClick={() => navigate("/")}>Home</button>
          <button onClick={() => navigate("/userdetails")}>Profile</button>
          <button onClick={() => navigate("/about")}>About this App</button>
        </div>
        <div className="navbar-right">
          <button onClick={handleLogout}>Logout</button>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <h3>Diarization Page: Need to be logged in to access this page</h3>
      </div>
    </div>
  );
};

export default MainPage;