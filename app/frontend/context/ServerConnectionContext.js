import React, { createContext, useContext, useEffect, useState, useRef } from 'react';
import classificationService from '../services/classificationService';

//this context will be used to share the connection status of t
const ServerConnectionContext = createContext();

export const ServerConnectionProvider = ({ children }) => {
    const [isConnected, setIsConnected] = useState(false);
    const [isChecking, setIsChecking] = useState(true);
    const retryConnectionRef = useRef(null);

    //checks for backend connection
    const checkConnection = async () => {
        setIsChecking(true);
        try {
            await classificationService.healthCheck();
            setIsConnected(true);
        }
        catch {
            setIsConnected(false);
        }
        finally {
            setIsChecking(false);
        }
    };

    //auto-retry if failed to connect
    useEffect(() => {
        if (!isConnected) {
            if (!retryConnectionRef.current) {
                retryConnectionRef.current = setInterval(checkConnection, 30000);
                console.log('Failed to connect. Trying to reconnect...');
            }
        } else {
            if (!retryConnectionRef.current) {
                clearInterval(retryConnectionRef.current);
            }
            retryConnectionRef.current = null;
            console.log('Connected successfully');
        }

        return () => {
            if (retryConnectionRef.current) {
                clearInterval(retryConnectionRef.current);
                retryConnectionRef.current = null;
            };
        }
    }, [isConnected]);

    useEffect(() => {
        checkConnection();
    }, []);

    return (
        <ServerConnectionContext.Provider
            value={{
                isConnected,
                isChecking,
                refreshConnection: checkConnection,
            }}
        >
            {children}
        </ServerConnectionContext.Provider>
    );
}

export const useServerConnection = () => useContext(ServerConnectionContext);