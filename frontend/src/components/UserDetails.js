import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { jwtDecode } from 'jwt-decode';
import './UserDetails.css';
import Navbar from './Navbar';

function UserDetails() {
  const [userDetails, setUserDetails] = useState(null);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const fetchUserDetails = useCallback(async (token) => {
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/auth/users/me/', {
        method: 'GET',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        if (response.status === 401) {
          throw new Error('Unauthorized. Please log in.');
        }
        throw new Error('Failed to fetch user details');
      }

      const data = await response.json();
      setUserDetails(data);
    } catch (err) {
      setError(err.message);
      if (err.message.includes('Unauthorized')) {
        navigate('/login');
      }
    }
  }, [navigate]);

  // const isTokenExpired = (token) => {
  //   try {
  //     const { exp } = jwtDecode(token);
  //     return Date.now() >= exp * 1000;
  //   } catch (error) {
  //     return true;
  //   }
  // };

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
  //         return;
  //       }
  //       fetchUserDetails(newAccessToken);
  //     });
  //   } else {
  //     fetchUserDetails(accessToken);
  //   }
  // }, [navigate, fetchUserDetails]);

  const handleLogout = () => {
    localStorage.removeItem('access');
    localStorage.removeItem('refresh');
    navigate('/login');
  };

  return (
    <div>
      <Navbar/>
      {/* <div className="navbar">
        <div className="navbar-left">
          <button onClick={() => navigate("/")}>Home</button>
          <button onClick={() => navigate("/diarization")}>Diarization</button>
          <button onClick={() => navigate("/about")}>About this App</button>
        </div>

        <div className="navbar-right">
          <button onClick={handleLogout}>Logout</button>
        </div>
      </div> */}

      <div className="userdetails-container font-play">
        <h2>User Details</h2>
        {error && <p className="error">{error}</p>}
        {userDetails && (
          <div>
            <p>Username: {userDetails.username}</p>
            <p>Email: {userDetails.email}</p>
            <p>First Name: {userDetails.first_name}</p>
            <p>Last Name: {userDetails.last_name}</p>
            <p>Phone Number: {userDetails.phone_number}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default UserDetails;