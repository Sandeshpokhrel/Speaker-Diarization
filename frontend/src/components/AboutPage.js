import React, { useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { jwtDecode } from "jwt-decode";
import "./AboutPage.css";

const AboutPage = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // // Function to check if a token is expired
  // const isTokenExpired = (token) => {
  //   try {
  //     const { exp } = jwtDecode(token);
  //     return Date.now() >= exp * 1000;
  //   } catch (error) {
  //     return true;
  //   }
  // };

  // // Function to refresh the access token
  // const refreshAccessToken = async (refreshToken) => {
  //   try {
  //     const response = await fetch("http://localhost:8000/auth/jwt/refresh", {
  //       method: "POST",
  //       headers: { "Content-Type": "application/json" },
  //       body: JSON.stringify({ refresh: refreshToken }),
  //     });

  //     if (!response.ok) {
  //       throw new Error("Failed to refresh access token");
  //     }

  //     const data = await response.json();
  //     localStorage.setItem("access", data.access);
  //     return data.access;
  //   } catch (error) {
  //     console.error("Error refreshing token:", error);
  //     return null;
  //   }
  // };

  // useEffect(() => {
  //   const accessToken = localStorage.getItem("access");
  //   const refreshToken = localStorage.getItem("refresh");

  //   // Redirect to login if tokens are missing
  //   if (!accessToken || !refreshToken) {
  //     navigate("/login");
  //     return;
  //   }

  //   // Check if the access token is expired
  //   if (isTokenExpired(accessToken)) {
  //     refreshAccessToken(refreshToken).then((newAccessToken) => {
  //       if (!newAccessToken) {
  //         navigate("/login");
  //       }
  //     });
  //   }
  // }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem("access");
    localStorage.removeItem("refresh");
    navigate("/login");
  };

  return (
    <div className="about-container">
      {/* Navbar */}
      <div className="navbar">
        <div className="navbar-left">
          <button
            className={`navbar-button ${location.pathname === '/' ? 'active' : ''}`}
            onClick={() => navigate('/')}
          >
            Home
          </button>
          <button
            className={`navbar-button ${location.pathname === '/diarization' ? 'active' : ''}`}
            onClick={() => navigate('/diarization')}
          >
            Diarization
          </button>
          <button
            className={`navbar-button ${location.pathname === '/userdetails' ? 'active' : ''}`}
            onClick={() => navigate('/userdetails')}
          >
            Profile
          </button>
        </div>
        <div className="navbar-right">
          <button className="logout-button" onClick={handleLogout}>
            Logout
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <h2 className="about-title">About This App</h2>
        <p>
          <strong>Welcome to the Future of Speaker Identification and Audio Analysis!</strong>
        </p>
        <p>
          Our app is designed to make it easy to identify and separate different speakers in audio recordings. Whether you're analyzing a meeting, an interview, or any group discussion, this app ensures accurate and efficient speaker tracking to save you time and effort.
        </p>

        <h3>Key Features:</h3>
        <ul>
          <li><strong>Speaker Separation:</strong> Automatically detect and label who is speaking in your audio files, even in complex multi-speaker scenarios.</li>
          <li><strong>Time-Stamped Segments:</strong> Get precise timestamps for each speaker, making it easy to review or transcribe audio content.</li>
          <li><strong>Accurate Results:</strong> Powered by advanced AI, our app delivers highly reliable results, even with overlapping or noisy audio.</li>
        </ul>

        <h3>Who Is It For?</h3>
        <ul>
          <li><strong>Businesses:</strong> Ideal for teams needing clear records of meetings or conferences.</li>
          <li><strong>Journalists and Researchers:</strong> Perfect for analyzing interviews or group discussions with multiple speakers.</li>
          <li><strong>Educators:</strong> A valuable tool for reviewing classroom recordings or group activities.</li>
        </ul>
      </div>
    </div>
  );
};

export default AboutPage;