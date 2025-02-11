import Navbar from "./Navbar";
import Pokhrel from '../sandesh.jpg'
import Pyakurel from '../pyakurel.jpg'
import Samip from '../samip.png'
const teamMembers = [
    {
      name: "Samip Neupane",
      role: "077BCT073",
      image: Samip,
    },
    {
      name: "Sandesh Pokhrel",
      role: "077BCT074",
      image: Pokhrel,
    },
    {
      name: "Sandesh Pyakurel",
      role: "077BCT075",
      image: Pyakurel,
    },
  ];
  
  export default function TeamPage() {
    return (
    <>
    <Navbar/>
        <div className="min-h-screen flex flex-col items-center justify-center bg-#0a0e16 p-8 font-play">
        <h2 className="text-3xl font-bold text-yellow-500 text-center mb-6">
          Meet Our Team
        </h2>
        <p className="text-gray-300 text-lg text-center mb-8">
          The brilliant minds behind this project.
        </p>
  
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-10">
          {teamMembers.map((member, index) => (
            <div
              key={index}
              className="w-80 h-96 p-8 bg-white/10 backdrop-blur-lg shadow-lg rounded-2xl border border-white/20 text-white flex flex-col items-center space-y-4"
            >
              <img
                src={member.image}
                alt={member.name}
                className="w-32 h-32 rounded-full border-2 border-blue-400 shadow-md"
              />
              <h3 className="text-2xl font-semibold text-yellow-500">
                {member.name}
              </h3>
              <p className="text-lg text-gray-300">{member.role}</p>
            </div>
          ))}
        </div>
      </div>
    </>
      
    );
  }
  