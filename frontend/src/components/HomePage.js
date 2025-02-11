import React from "react";
import { useNavigate } from "react-router-dom";
import "./HomePage.css";
import background from "../background.gif" 
const HomePage = () => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate("/overviewPage");
  };

  return (
//     <div className="home-page">
//       <div className="text-section">
//       <h1 className="app-title">SPEAKER DIARIZATION</h1>
// <h2 className="app-subtitle">
//   Easily identify and separate different speakers in your audio.
// </h2>
// <p className="app-description">
//   Our app helps you break down audio recordings by identifying who is speaking and when. Perfect for meetings, interviews, or any group discussions, it makes it simple to track and label speakers. With advanced technology, you get accurate results that save time and make audio analysis easy.
// </p>

//         <button className="get-started-button" onClick={handleGetStarted}>
//           GET STARTED →
//         </button>
//       </div>
//       <div className="auth-buttons">
//         <a href="/login" className="auth-btn login-btn">
//           Login
//         </a>
//         <a href="/register" className="auth-btn register-btn">
//           Register
//         </a>
//       </div>
//     </div>
<React.Fragment>
  <div className="relative w-full h-screen overflow-hidden"> {/* Ensure the container covers the full screen height */}
  <div className="relative z-10 h-screen flex flex-col justify-center items-center text-center">
  <h1 className="text-4xl font-bold sm:text-5xl">
    Classify The Audio With Our
  </h1>
  <h2 className="text-5xl font-bold text-primary mt-2">
    Nepali Speaker Diarizer
  </h2>
  <p className="pt-3 text-lg lg:w-3/5 lg:pt-5">
    Experience the epitome of elegance and precision with our exquisite
    range of luxury timepieces. Elevate every moment with unparalleled
    style.
  </p>
  <button className="mt-5 w-1/2 px-3 py-1 btn lg:h-10 lg:w-3/12 lg:px-10 hover:bg-gold-500 hover:text-white focus:outline-none focus-visible:outline-none focus:ring-0" onClick={handleGetStarted}>
    Explore
  </button>
</div>
    <img
      src={`${background}`}
      className="absolute inset-0 w-full h-full object-cover opacity-30 animate-fade-in"
      alt="Background"
    />
  </div>
</React.Fragment>
  );
};

export default HomePage;
