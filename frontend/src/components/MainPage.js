import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { jwtDecode } from "jwt-decode";
import "./MainPage.css";

const MainPage = () => {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [diarizationResults, setDiarizationResults] = useState(null);

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

  //   if (!accessToken || !refreshToken) {
  //     navigate("/login");
  //     return;
  //   }

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

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    setDiarizationResults([
      ["speaker_1", [0.00, 5.35, "नमस्कार! पुल्चोक कताबाट जान्छ जानकारी गरिदिनुस न।"]],
      ["speaker_2", [6.10, 8.43, "पुल्चोक रत्नपार्कबाट गाडी चढेर जना मिल्छ।"]],
      ["speaker_0", [9.44, 14.56, "तपाई हिड्दै गएपनि आधि घण्टामा मज्जाले पुग्नुहुन्छ।"]],
      ["speaker_1", [15.00, 17.85, "धेरै धेरै धन्यवाद!"]],
      ["speaker_3", [20.09, 24.35, "पर्खनुस त, म पनि जादै छु पुल्चोक।"]],
      ["speaker_2", [25.06, 30.39, "ल हजुरले साथी भेट्नुभयो, कुरा गर्दै जानुहोस।"]],
      ["speaker_0", [31.22, 33.88, "राम्रोसंग जानुहोला।"]],
      ["speaker_1", [33.90, 37.45, "हुन्छ, मौका मिल्यो भने फेरी भेट्न पाईएला।"]]
    ]);
  };

  const speakerColors = {
    'speaker_0': '#8884d8',
    'speaker_1': '#82ca9d',
    'speaker_2': '#ffc658',
    'speaker_3': '#ff8042'
  };

  return (
    <div className="second-container">
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
  
      <div className="main-content">
        <h1 className="center-heading">Speakers Diarization</h1>
        
        <div className="upload-section">
          <form onSubmit={handleSubmit}>
            <div className="file-input-container">
              <label className="upload-label">Upload Audio File</label>
              <input
                type="file"
                accept="audio/*"
                onChange={handleFileChange}
                className="file-input"
              />
            </div>
            <button
              type="submit"
              className="action-button"
              disabled={!selectedFile}
            >
              Process Audio
            </button>
          </form>
        </div>
  
        {diarizationResults && (
          <div className="results-container">
            <div className="compact-results">
              <h3 className="center-heading">Diarization Output</h3>
              <table className="transcript-table">
                <thead>
                  <tr>
                    <th>Speaker</th>
                    <th>Time</th>
                    <th>Transcript</th>
                  </tr>
                </thead>
                <tbody>
                  {diarizationResults.map(([speaker, [start, end, content]], index) => (
                    <tr key={index}>
                      <td>{speaker}</td>
                      <td>{start.toFixed(2)}s - {end.toFixed(2)}s</td>
                      <td className="transcript-content">{content}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );  
};

export default MainPage;