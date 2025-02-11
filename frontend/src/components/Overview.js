import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { FaMicrophoneAlt, FaDatabase, FaWaveSquare, FaBrain, FaChartLine } from 'react-icons/fa';
import Navbar from './Navbar';

const methodologies = [
  { title: 'Data Collection', description: 'Collected dataset using OpenSLR.', icon: <FaDatabase /> },
  { title: 'Preprocessing', description: 'Noise reduction, overlapping & merging.', icon: <FaWaveSquare /> },
  { title: 'Feature Extraction', description: 'Extracted MFCC features.', icon: <FaMicrophoneAlt /> },
  { title: 'Model Training', description: 'Trained EEND model on Kaldi format data.', icon: <FaBrain /> },
  { title: 'Evaluation', description: 'Calculated DER (Diarization Error Rate).', icon: <FaChartLine /> },
];

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <>
        <Navbar/>
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center p-10 font-play">
        
      {/* Title */}
      <motion.h1 
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="text-4xl font-bold text-center mb-8 text-yellow-500"
      >
        🔊 End-to-End Speaker Diarization
      </motion.h1>

      {/* Description */}
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
        className="text-lg text-center max-w-3xl text-gray-300"
      >
        An AI-powered system to separate and identify speakers in an audio stream using the EEND model.
      </motion.p>

      {/* Methodology Tiles */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-10 w-full max-w-5xl">
        {methodologies.map((method, index) => (
          <motion.div 
            key={index}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: index * 0.2 }}
            className="bg-gray-800 p-6 rounded-2xl flex flex-col items-center shadow-lg hover:bg-blue-600 transition-all duration-300 cursor-pointer"
          >
            <div className="text-4xl mb-3 text-yellow-400">{method.icon}</div>
            <h2 className="text-xl font-semibold">{method.title}</h2>
            <p className="text-gray-300 text-sm mt-2 text-center">{method.description}</p>
          </motion.div>
        ))}
      </div>

      {/* CTA Button */}
      <motion.button
        onClick={() => navigate('/diarization')}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 1 }}
        className="mt-10 px-6 py-3 bg-yellow-500 text-black font-bold rounded-xl shadow-md hover:bg-yellow-600 transition-all duration-300"
      >
        Try Speaker Diarization
      </motion.button>
    </div>
    </>
    
  );
}
