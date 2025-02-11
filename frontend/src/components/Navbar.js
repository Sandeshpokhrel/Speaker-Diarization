import { Disclosure, DisclosureButton, DisclosurePanel, Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';
import { Bars3Icon, BellIcon, XMarkIcon, UserIcon } from '@heroicons/react/24/outline';
import { useNavigate, useLocation } from 'react-router-dom';

const navigation = [
  { name: 'Overview', href: '/overviewPage' },
  { name: 'Demo', href: '/diarization' },
  { name: 'About', href: '/about' },
  { name: 'Team', href: '/teamsPage' },
];

function classNames(...classes) {
  return classes.filter(Boolean).join(' ');
}

export default function Navbar() {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <>
      {/* Fixed Navbar */}
      <Disclosure as="nav" className="fixed top-0 w-full z-50 bg-gray-800 font-play shadow-md">
        <div className="mx-auto max-w-7xl px-2 sm:px-6 lg:px-8">
          <div className="relative flex h-16 items-center justify-between">
            {/* Mobile Menu Button */}
            <div className="absolute inset-y-0 left-0 flex items-center sm:hidden">
              <DisclosureButton className="group relative inline-flex items-center justify-center rounded-md p-2 text-gray-400 hover:bg-gray-700 hover:text-white focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white">
                <Bars3Icon className="block size-6 group-data-[open]:hidden" />
                <XMarkIcon className="hidden size-6 group-data-[open]:block" />
              </DisclosureButton>
            </div>

            {/* Logo and Navigation */}
            <div className="flex flex-1 items-center justify-center sm:items-stretch sm:justify-start">
              {/* Brand Name */}
              <div
                className="flex shrink-0 items-center cursor-pointer text-yellow-500 text-xl font-semibold tracking-wide hover:text-blue-500 transition-all duration-300"
                onClick={() => navigate("/")}
              >
                <span>🔊 Speaker Diarizer</span>
              </div>

              {/* Navigation Links */}
              <div className="hidden sm:ml-8 sm:block">
                <div className="flex space-x-6">
                  {navigation.map((item) => (
                    <button
                      key={item.name}
                      onClick={() => navigate(item.href)}
                      className={classNames(
                        location.pathname === item.href ? 'bg-gray-900 text-white' : 'text-gray-300',
                        'rounded-lg px-4 py-3 text-base font-medium transition-all duration-300 group',
                        'hover:!bg-blue-900 hover:!text-white'
                      )}
                    >
                      {item.name}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Notifications and User Profile */}
            <div className="absolute inset-y-0 right-0 flex items-center pr-2 sm:static sm:inset-auto sm:ml-6 sm:pr-0">
              {/* Notification Bell */}
              <button
                type="button"
                className="relative rounded-full bg-gray-800 p-1 text-gray-400 hover:text-white focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-gray-800"
              >
                <BellIcon className="size-6" />
              </button>

              {/* User Profile Dropdown */}
              <Menu as="div" className="relative ml-3">
                <div>
                  <MenuButton className="relative flex items-center justify-center rounded-full bg-gray-800 p-2 text-sm text-gray-400 hover:text-white focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-gray-800 transition">
                    <UserIcon className="h-6 w-6" />
                  </MenuButton>
                </div>
                <MenuItems
                  transition
                  className="absolute right-0 z-10 mt-2 w-48 origin-top-right rounded-md bg-white py-1 shadow-lg ring-1 ring-black/5 transition focus:outline-none"
                >
                  <MenuItem>
                    <a href="#" className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">
                      Your Profile
                    </a>
                  </MenuItem>
                  <MenuItem>
                    <a href="#" className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">
                      Settings
                    </a>
                  </MenuItem>
                  <MenuItem>
                    <a href="/" className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">
                      Sign out
                    </a>
                  </MenuItem>
                </MenuItems>
              </Menu>
            </div>
          </div>
        </div>

        {/* Mobile Navigation Menu */}
        <DisclosurePanel className="sm:hidden">
          <div className="space-y-1 px-2 pb-3 pt-2">
            {navigation.map((item) => (
              <DisclosureButton
                key={item.name}
                onClick={() => navigate(item.href)}
                className={classNames(
                  location.pathname === item.href ? 'bg-gray-900 text-white' : 'text-gray-300',
                  'custom-nav-item block rounded-md px-3 py-2 text-base font-medium'
                )}
              >
                {item.name}
              </DisclosureButton>
            ))}
          </div>
        </DisclosurePanel>
      </Disclosure>

      {/* Spacer to push content below the fixed navbar */}
      <div className="pt-16"></div>
    </>
  );
}
