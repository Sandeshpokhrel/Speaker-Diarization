import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { jwtDecode } from "jwt-decode";
import "./MainPage.css";
import Navbar from "./Navbar";
import fetchFramework from "../framework/fetchFramework";
const MainPage = () => {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [diarizationResults, setDiarizationResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const handleLogout = () => {
    localStorage.removeItem("access");
    localStorage.removeItem("refresh");
    navigate("/login");
  };

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleSubmit = async(event) => {
    setIsLoading(true); // Set loading to true when submitting
    const formData = new FormData();
    formData.append("audio", selectedFile);
    event.preventDefault();
    // setDiarizationResults([
    //   ["speaker_1", [0.00, 5.35, "नमस्कार! पुल्चोक कताबाट जान्छ जानकारी गरिदिनुस न।"]],
    //   ["speaker_2", [6.10, 8.43, "पुल्चोक रत्नपार्कबाट गाडी चढेर जना मिल्छ।"]],
    //   ["speaker_0", [9.44, 14.56, "तपाई हिड्दै गएपनि आधि घण्टामा मज्जाले पुग्नुहुन्छ।"]],
    //   ["speaker_1", [15.00, 17.85, "धेरै धेरै धन्यवाद!"]],
    //   ["speaker_3", [20.09, 24.35, "पर्खनुस त, म पनि जादै छु पुल्चोक।"]],
    //   ["speaker_2", [25.06, 30.39, "ल हजुरले साथी भेट्नुभयो, कुरा गर्दै जानुहोस।"]],
    //   ["speaker_0", [31.22, 33.88, "राम्रोसंग जानुहोला।"]],
    //   ["speaker_1", [33.90, 37.45, "हुन्छ, मौका मिल्यो भने फेरी भेट्न पाईएला।"]]
    // ]);
    try {
      let response = await fetchFramework({endpoint: "/api/speaker_diarization/", form: formData});
      console.log(response.diarization_result);
      setDiarizationResults(response.diarization_result);
    }
    catch (error) {
      console.error(error);
    }finally {
      setIsLoading(false); // Set loading to false when done
    }
  };
 
  const speakerColors = {
    'speaker_0': '#8884d8',
    'speaker_1': '#82ca9d',
    'speaker_2': '#ffc658',
    'speaker_3': '#ff8042'
  };

  return (
    <div className="second-container font-play">
      <Navbar/>
      {/* <div className="navbar"> */}
        {/* <div className="navbar-left">
          <button onClick={() => navigate("/")}>Home</button>
          <button onClick={() => navigate("/userdetails")}>Profile</button>
          <button onClick={() => navigate("/about")}>About this App</button>
        </div>
        <div className="navbar-right">
          <button onClick={handleLogout}>Logout</button>
        </div> */}
        
      {/* </div> */}
      <div className="main-content ">
      
        <h1 className="text-yellow-500 text-3xl font-bold center-heading">Speakers Diarization</h1>
        
        <div className="flex items-center justify-center bg-gradient-to-r ">
          <div className="p-6 w-96 bg-white/10 backdrop-blur-lg shadow-lg rounded-2xl border border-white/20">
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* File Upload Section */}
              <div className="flex flex-col items-center gap-3">
                <label className="text-white text-lg font-semibold">
                  Upload Audio File
                </label>
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer bg-white/20 text-white py-2 px-4 rounded-lg transition-all duration-300 hover:bg-white/30"
                >
                  Choose File
                </label>
                {selectedFile && (
                  <p className="text-sm text-gray-300">Selected: {selectedFile.name}</p>
                )}
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                className="w-full bg-blue-500 text-white font-semibold py-2 px-4 rounded-lg shadow-md transition-all duration-300 hover:bg-blue-600 disabled:opacity-50"
                disabled={!selectedFile || isLoading}
                onClick={handleSubmit}
              >
                {isLoading ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full mr-2"></div>
                  Processing...
                </div>
              ) : (
                'Process Audio'
              )}
              </button>
            </form>
          </div>
        </div>

  
        {diarizationResults && (
          <div className="results-container">
            <div className="compact-results">
              <h3 className="center-heading">Diarization Output</h3>
              <div className="table-wrapper">
              <table className="w-full border-collapse rounded-lg overflow-hidden shadow">
                  <thead className="bg-blue-600 text-white">
                    <tr>
                      <th className="py-3 px-4 text-center">Speaker</th>
                      <th className="py-3 px-4 text-center w-1/4">Time</th> {/* Wider Time Column */}
                      <th className="py-3 px-4 text-center">Transcript</th>
                    </tr>
                  </thead>
                  <tbody>
                    {diarizationResults.map(([speaker, start, end, text], index) => (
                      <tr
                        key={index}
                        className={`${
                          index % 2 === 0 ? "bg-gray-100" : "bg-white"
                        } hover:bg-blue-50 transition`}
                      >
                        <td className="py-3 px-4 font-medium text-blue-700">{speaker}</td>
                        <td className="py-3 px-4 italic text-gray-600">
                          {start.toFixed(2)}s - {end.toFixed(2)}s
                        </td>
                        <td className="py-3 px-4 text-gray-800">{text}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
      
  
      
    </div>
  );  
};

export default MainPage;