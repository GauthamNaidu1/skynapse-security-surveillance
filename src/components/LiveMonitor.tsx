import React, { useState, useEffect } from 'react';
import { MapPin, Clock, Shield, Siren, Radio } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Custom Dot Icons
const createDotIcon = (color: string) => L.divIcon({
  className: 'custom-dot-icon',
  html: `<div style="background-color: ${color}; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 4px rgba(0,0,0,0.5);"></div>`,
  iconSize: [12, 12],
  iconAnchor: [6, 6],
  popupAnchor: [0, -6]
});

const icons = {
  Critical: createDotIcon('#ef4444'), // Red-500
  High: createDotIcon('#f97316'),     // Orange-500
  Medium: createDotIcon('#eab308'),   // Yellow-500
  Low: createDotIcon('#3b82f6'),      // Blue-500
  Default: createDotIcon('#64748b')   // Slate-500
};

interface Incident {
  id: number;
  timestamp: string;
  location: string;
  type: string;
  severity: string;
  lat: number;
  lng: number;
}

export const DEFAULT_HYDERABAD_INCIDENTS: Incident[] = [
  { id: 1, type: "Theft", location: "Charminar", severity: "Medium", lat: 17.3616, lng: 78.4747, timestamp: new Date(Date.now() - 7200000).toISOString() },
  { id: 2, type: "Traffic Violation", location: "Hitech City", severity: "Low", lat: 17.4435, lng: 78.3772, timestamp: new Date(Date.now() - 10800000).toISOString() },
  { id: 3, type: "Fighting", location: "Secunderabad", severity: "High", lat: 17.4399, lng: 78.4983, timestamp: new Date(Date.now() - 14400000).toISOString() },
  { id: 4, type: "Suspicious Activity", location: "Banjara Hills", severity: "Low", lat: 17.4138, lng: 78.4397, timestamp: new Date(Date.now() - 18000000).toISOString() },
  { id: 5, type: "Chain Snatching", location: "Kukatpally", severity: "Critical", lat: 17.4849, lng: 78.3968, timestamp: new Date(Date.now() - 21600000).toISOString() },
  { id: 6, type: "Vandalism", location: "Ameerpet", severity: "Medium", lat: 17.4375, lng: 78.4483, timestamp: new Date(Date.now() - 25200000).toISOString() },
  { id: 7, type: "Theft", location: "Madhapur", severity: "Medium", lat: 17.4484, lng: 78.3908, timestamp: new Date(Date.now() - 28800000).toISOString() },
  { id: 8, type: "Traffic Violation", location: "Gachibowli", severity: "Low", lat: 17.4401, lng: 78.3489, timestamp: new Date(Date.now() - 32400000).toISOString() },
  { id: 9, type: "Robbery", location: "Dilsukhnagar", severity: "High", lat: 17.3687, lng: 78.5247, timestamp: new Date(Date.now() - 36000000).toISOString() },
  { id: 10, type: "Fighting", location: "Malakpet", severity: "High", lat: 17.3763, lng: 78.4983, timestamp: new Date(Date.now() - 39600000).toISOString() },
  { id: 11, type: "Theft", location: "Begumpet", severity: "Low", lat: 17.4440, lng: 78.4674, timestamp: new Date(Date.now() - 43200000).toISOString() },
  { id: 12, type: "Traffic Violation", location: "LB Nagar", severity: "Low", lat: 17.3457, lng: 78.5522, timestamp: new Date(Date.now() - 46800000).toISOString() },
  { id: 13, type: "Suspicious Activity", location: "Jubilee Hills", severity: "Low", lat: 17.4325, lng: 78.4070, timestamp: new Date(Date.now() - 50400000).toISOString() },
  { id: 14, type: "Theft", location: "Abids", severity: "Medium", lat: 17.3927, lng: 78.4758, timestamp: new Date(Date.now() - 54000000).toISOString() },
  { id: 15, type: "Chain Snatching", location: "Mehdipatnam", severity: "Critical", lat: 17.3950, lng: 78.4422, timestamp: new Date(Date.now() - 57600000).toISOString() },
  { id: 16, type: "Theft", location: "Uppal", severity: "Medium", lat: 17.4065, lng: 78.5593, timestamp: new Date(Date.now() - 61200000).toISOString() },
  { id: 17, type: "Traffic Violation", location: "Miyapur", severity: "Low", lat: 17.4969, lng: 78.3548, timestamp: new Date(Date.now() - 64800000).toISOString() },
  { id: 18, type: "Theft", location: "Kondapur", severity: "Medium", lat: 17.4625, lng: 78.3522, timestamp: new Date(Date.now() - 68400000).toISOString() },
  { id: 19, type: "Fighting", location: "Tarnaka", severity: "High", lat: 17.4260, lng: 78.5304, timestamp: new Date(Date.now() - 72000000).toISOString() },
  { id: 20, type: "Fighting", location: "Nampally", severity: "High", lat: 17.3866, lng: 78.4678, timestamp: new Date(Date.now() - 75600000).toISOString() }
];

// Map Controller to handle flying to location
function MapController({ selectedLocation }: { selectedLocation: { lat: number, lng: number } | null }) {
  const map = useMap();
  useEffect(() => {
    if (selectedLocation) {
      map.flyTo([selectedLocation.lat, selectedLocation.lng], 15, {
        duration: 1.5
      });
    }
  }, [selectedLocation, map]);
  return null;
}

export default function LiveMonitor() {
  const [incidents, setIncidents] = useState<Incident[]>(DEFAULT_HYDERABAD_INCIDENTS);
  const [loading, setLoading] = useState(false);
  const [selectedIncidentId, setSelectedIncidentId] = useState<number | null>(null);
  const [totalIncidents, setTotalIncidents] = useState(DEFAULT_HYDERABAD_INCIDENTS.length);
  const [feedMode, setFeedMode] = useState<'historical' | 'simulation'>('historical');

  useEffect(() => {
    if (feedMode === 'historical') {
      // Load real/baseline historical incidents from SQLite
      fetch('/api/history?mode=historical')
        .then(res => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then(data => {
          if (Array.isArray(data) && data.length > 0) {
            setIncidents(data);
            setTotalIncidents(data.length);
          } else {
            // Keep default baseline if API returns empty
            setIncidents(DEFAULT_HYDERABAD_INCIDENTS);
            setTotalIncidents(DEFAULT_HYDERABAD_INCIDENTS.length);
          }
          setLoading(false);
        })
        .catch(err => {
          console.warn("Using baseline historical incidents fallback:", err);
          setIncidents(DEFAULT_HYDERABAD_INCIDENTS);
          setTotalIncidents(DEFAULT_HYDERABAD_INCIDENTS.length);
          setLoading(false);
        });
      return;
    }

    // Demo Simulation Mode: periodically generates and records simulated incidents
    const fetchSimulatedIncident = async () => {
      try {
        const res = await fetch('/api/live-incident?mode=simulation');
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();
        if (!data || !data.id || !data.type) return;

        setIncidents(prev => {
          if (prev.find(i => i.id === data.id)) return prev;
          setTotalIncidents(c => c + 1);
          return [data, ...prev].slice(0, 50);
        });
        setLoading(false);
      } catch (err) {
        console.error("Simulation feed error:", err);
        setLoading(false);
      }
    };

    fetchSimulatedIncident();
    const interval = setInterval(fetchSimulatedIncident, 4000);
    return () => clearInterval(interval);
  }, [feedMode]);

  const selectedIncident = incidents.find(i => i.id === selectedIncidentId);

  return (
    <div className="p-6 space-y-6 h-full overflow-y-auto bg-slate-950 text-slate-200">
      <header className="flex justify-between items-center border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
              <Radio className={`w-5 h-5 ${feedMode === 'simulation' ? 'text-amber-500 animate-pulse' : 'text-blue-500'}`} />
              Incident Geospatial Monitor
            </h2>
            {/* Visible Mode Badge */}
            <span className={`px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold border ${
              feedMode === 'historical'
                ? 'bg-blue-950/60 border-blue-600/40 text-blue-400'
                : 'bg-amber-950/60 border-amber-600/40 text-amber-400'
            }`}>
              {feedMode === 'historical' ? 'Historical data' : 'Simulated feed'}
            </span>
          </div>
          <p className="text-sm text-slate-500 mt-1">
            {feedMode === 'historical'
              ? 'Displaying verified historical incident records from SQLite database'
              : 'Interactive synthetic demonstration stream for UI evaluation'}
          </p>
        </div>

        <div className="flex items-center gap-4">
          {/* Mode Switcher */}
          <div className="flex bg-slate-900 p-1 rounded-lg border border-slate-800">
            <button
              onClick={() => setFeedMode('historical')}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                feedMode === 'historical'
                  ? 'bg-blue-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Historical Data
            </button>
            <button
              onClick={() => setFeedMode('simulation')}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                feedMode === 'simulation'
                  ? 'bg-amber-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Demo Simulation
            </button>
          </div>

          <div className="flex flex-col items-end pl-2">
            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Total Records</span>
            <span className="text-2xl font-bold text-white leading-none font-mono">{totalIncidents}</span>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Incident List */}
        <div className="lg:col-span-1 space-y-4 h-[600px] overflow-y-auto pr-2">
          <AnimatePresence mode='popLayout'>
            {incidents.map((incident, index) => (
              <motion.div
                key={incident.id}
                initial={{ opacity: 0, x: -20, scale: 0.95 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                layout
                onClick={() => setSelectedIncidentId(incident.id)}
                className={`relative overflow-hidden rounded-xl border p-4 transition-all cursor-pointer group ${
                  selectedIncidentId === incident.id
                    ? 'bg-slate-800 border-slate-600 shadow-lg shadow-blue-900/20 ring-1 ring-slate-500'
                    : incident.severity === 'Critical' 
                      ? 'bg-red-950/20 border-red-900/50 hover:bg-red-950/30' 
                      : 'bg-slate-900/50 border-slate-800 hover:bg-slate-800'
                }`}
              >
                <div className="flex justify-between items-start relative z-10">
                  <div className="flex items-start gap-3">
                    <div className={`p-2 rounded-lg ${
                      incident.severity === 'Critical' ? 'bg-red-500/20 text-red-500' : 'bg-slate-800 text-slate-400'
                    }`}>
                      {incident.severity === 'Critical' ? <Siren className="w-5 h-5 animate-bounce" /> : <Shield className="w-5 h-5" />}
                    </div>
                    <div>
                      <h3 className="font-semibold text-white text-sm">{incident.type}</h3>
                      <div className="flex flex-col gap-1 mt-1 text-xs text-slate-400">
                        <span className="flex items-center gap-1">
                          <MapPin className="w-3 h-3" /> {incident.location}
                        </span>
                        <span className="flex items-center gap-1 font-mono opacity-70">
                          <Clock className="w-3 h-3" /> {new Date(incident.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="text-xs font-mono text-slate-500 font-bold pt-1 opacity-50">
                    #{incidents.length - index}
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          
          {loading && (
            <div className="flex justify-center p-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-red-500"></div>
            </div>
          )}
        </div>

        {/* Map Visualization */}
        <div className="lg:col-span-2 bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden h-[600px] relative z-0">
            <MapContainer 
                center={[17.3850, 78.4867]} 
                zoom={13} 
                style={{ height: '100%', width: '100%' }}
                className="z-0"
            >
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    className="dark-tiles"
                    maxZoom={19}
                />
                
                <MapController selectedLocation={selectedIncident ? { lat: selectedIncident.lat, lng: selectedIncident.lng } : null} />

                {/* Incident Markers */}
                {incidents.map((incident) => (
                    <Marker 
                        key={incident.id} 
                        position={[incident.lat, incident.lng]}
                        icon={icons[incident.severity as keyof typeof icons] || icons.Default}
                        eventHandlers={{
                          click: () => {
                            setSelectedIncidentId(incident.id);
                          },
                        }}
                    >
                        <Popup className="custom-popup">
                            <div className="text-slate-900 min-w-[150px]">
                                <strong className="text-sm block mb-1">{incident.type}</strong>
                                <div className="text-xs text-slate-600 mb-2">{incident.location}</div>
                                <span className={`text-[10px] font-bold px-2 py-1 rounded-full text-white ${
                                  incident.severity === 'Critical' ? 'bg-red-600' : 
                                  incident.severity === 'High' ? 'bg-orange-500' :
                                  incident.severity === 'Medium' ? 'bg-yellow-500' : 'bg-blue-500'
                                }`}>
                                    {incident.severity.toUpperCase()}
                                </span>
                            </div>
                        </Popup>
                    </Marker>
                ))}
            </MapContainer>
            
            <div className="absolute top-4 right-4 bg-slate-900/80 backdrop-blur p-2 rounded border border-slate-700 z-[1000]">
                <div className="text-xs font-mono text-slate-300 mb-1">MAP LAYERS</div>
                <div className="flex items-center gap-2 text-xs text-slate-400">
                    <span className="w-2 h-2 rounded-full bg-blue-500"></span> Incidents
                </div>
            </div>
        </div>
      </div>
    </div>
  );
}
