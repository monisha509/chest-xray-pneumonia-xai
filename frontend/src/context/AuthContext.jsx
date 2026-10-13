import React, { createContext, useContext, useState } from 'react';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    return sessionStorage.getItem('chestxr_session_user') || null;
  });

  const login = (username) => {
    sessionStorage.setItem('chestxr_session_user', username);
    setUser(username);
  };

  const logout = () => {
    sessionStorage.removeItem('chestxr_session_user');
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

