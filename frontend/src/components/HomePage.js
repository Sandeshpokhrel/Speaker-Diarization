import React from "react";
import { useNavigate } from "react-router-dom";
import "./HomePage.css";

const HomePage = () => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate("/diarization");
  };

  return (
    <div className="home-page">
      <div className="text-section">
      <h1 className="app-title">SPEAKER DIARIZATION</h1>
<h2 className="app-subtitle">
  Easily identify and separate different speakers in your audio.
</h2>
<p className="app-description">
  Our app helps you break down audio recordings by identifying who is speaking and when. Perfect for meetings, interviews, or any group discussions, it makes it simple to track and label speakers. With advanced technology, you get accurate results that save time and make audio analysis easy.
</p>

        <button className="get-started-button" onClick={handleGetStarted}>
          GET STARTED →
        </button>
      </div>
      <div className="auth-buttons">
        <a href="/login" className="auth-btn login-btn">
          Login
        </a>
        <a href="/register" className="auth-btn register-btn">
          Register
        </a>
      </div>
    </div>
  );
};

export default HomePage;
