import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dbPath = path.join(__dirname, '../../skynapse.db');

// Coordinate mapping for all 20 Hyderabad monitored areas
export const LOCATIONS_COORDS: Record<string, { lat: number; lng: number }> = {
  "Hitech City": { lat: 17.4435, lng: 78.3772 },
  "Charminar": { lat: 17.3616, lng: 78.4747 },
  "Banjara Hills": { lat: 17.4138, lng: 78.4397 },
  "Secunderabad": { lat: 17.4399, lng: 78.4983 },
  "Gachibowli": { lat: 17.4401, lng: 78.3489 },
  "Jubilee Hills": { lat: 17.4325, lng: 78.4070 },
  "Ameerpet": { lat: 17.4375, lng: 78.4483 },
  "LB Nagar": { lat: 17.3457, lng: 78.5522 },
  "Madhapur": { lat: 17.4484, lng: 78.3908 },
  "Begumpet": { lat: 17.4440, lng: 78.4674 },
  "Kukatpally": { lat: 17.4849, lng: 78.3968 },
  "Dilsukhnagar": { lat: 17.3687, lng: 78.5247 },
  "Mehdipatnam": { lat: 17.3950, lng: 78.4422 },
  "Uppal": { lat: 17.4065, lng: 78.5593 },
  "Miyapur": { lat: 17.4969, lng: 78.3548 },
  "Kondapur": { lat: 17.4625, lng: 78.3522 },
  "Tarnaka": { lat: 17.4260, lng: 78.5304 },
  "Abids": { lat: 17.3927, lng: 78.4758 },
  "Nampally": { lat: 17.3866, lng: 78.4678 },
  "Malakpet": { lat: 17.3763, lng: 78.4983 }
};

// Comprehensive reference baseline covering all 20 Hyderabad locations
export const BASELINE_INCIDENTS = [
  { type: "Theft", location: "Charminar", severity: "Medium", hoursAgo: 2 },
  { type: "Chain Snatching", location: "Charminar", severity: "Critical", hoursAgo: 14 },
  { type: "Traffic Violation", location: "Hitech City", severity: "Low", hoursAgo: 3 },
  { type: "Suspicious Activity", location: "Hitech City", severity: "Low", hoursAgo: 18 },
  { type: "Fighting", location: "Secunderabad", severity: "High", hoursAgo: 4 },
  { type: "Theft", location: "Secunderabad", severity: "Medium", hoursAgo: 22 },
  { type: "Suspicious Activity", location: "Banjara Hills", severity: "Low", hoursAgo: 5 },
  { type: "Vandalism", location: "Banjara Hills", severity: "Medium", hoursAgo: 26 },
  { type: "Traffic Violation", location: "Gachibowli", severity: "Low", hoursAgo: 6 },
  { type: "Theft", location: "Gachibowli", severity: "Low", hoursAgo: 30 },
  { type: "Robbery", location: "Jubilee Hills", severity: "High", hoursAgo: 7 },
  { type: "Suspicious Activity", location: "Jubilee Hills", severity: "Low", hoursAgo: 32 },
  { type: "Vandalism", location: "Ameerpet", severity: "Medium", hoursAgo: 8 },
  { type: "Fighting", location: "Ameerpet", severity: "High", hoursAgo: 36 },
  { type: "Traffic Violation", location: "LB Nagar", severity: "Low", hoursAgo: 9 },
  { type: "Robbery", location: "LB Nagar", severity: "High", hoursAgo: 40 },
  { type: "Theft", location: "Madhapur", severity: "Medium", hoursAgo: 10 },
  { type: "Traffic Violation", location: "Madhapur", severity: "Low", hoursAgo: 42 },
  { type: "Suspicious Activity", location: "Begumpet", severity: "Low", hoursAgo: 11 },
  { type: "Theft", location: "Begumpet", severity: "Medium", hoursAgo: 45 },
  { type: "Chain Snatching", location: "Kukatpally", severity: "Critical", hoursAgo: 12 },
  { type: "Fighting", location: "Kukatpally", severity: "High", hoursAgo: 48 },
  { type: "Robbery", location: "Dilsukhnagar", severity: "High", hoursAgo: 13 },
  { type: "Theft", location: "Dilsukhnagar", severity: "Medium", hoursAgo: 50 },
  { type: "Chain Snatching", location: "Mehdipatnam", severity: "Critical", hoursAgo: 15 },
  { type: "Traffic Violation", location: "Mehdipatnam", severity: "Low", hoursAgo: 52 },
  { type: "Theft", location: "Uppal", severity: "Medium", hoursAgo: 16 },
  { type: "Fighting", location: "Uppal", severity: "High", hoursAgo: 55 },
  { type: "Traffic Violation", location: "Miyapur", severity: "Low", hoursAgo: 17 },
  { type: "Suspicious Activity", location: "Miyapur", severity: "Low", hoursAgo: 58 },
  { type: "Theft", location: "Kondapur", severity: "Medium", hoursAgo: 19 },
  { type: "Traffic Violation", location: "Kondapur", severity: "Low", hoursAgo: 60 },
  { type: "Fighting", location: "Tarnaka", severity: "High", hoursAgo: 20 },
  { type: "Suspicious Activity", location: "Tarnaka", severity: "Low", hoursAgo: 62 },
  { type: "Theft", location: "Abids", severity: "Medium", hoursAgo: 21 },
  { type: "Robbery", location: "Abids", severity: "High", hoursAgo: 65 },
  { type: "Fighting", location: "Nampally", severity: "High", hoursAgo: 23 },
  { type: "Traffic Violation", location: "Nampally", severity: "Low", hoursAgo: 68 },
  { type: "Fighting", location: "Malakpet", severity: "High", hoursAgo: 24 },
  { type: "Robbery", location: "Malakpet", severity: "High", hoursAgo: 70 }
];

/**
 * Pure JavaScript In-Memory Database Engine
 * Zero native binary dependencies, runs everywhere (Vercel Serverless, Node, Edge).
 */
class InMemoryDatabase {
  incidents: any[] = [];
  predictions: any[] = [];
  training_logs: any[] = [];
  nextIncidentId = 1;
  nextPredictionId = 1;

  constructor() {
    this.seedDefaultIncidents();
  }

  seedDefaultIncidents() {
    this.incidents = BASELINE_INCIDENTS.map((inc, i) => {
      const coords = LOCATIONS_COORDS[inc.location] || { lat: 17.3850, lng: 78.4867 };
      return {
        id: this.nextIncidentId++,
        type: inc.type,
        location: inc.location,
        lat: coords.lat + ((i % 5 - 2) * 0.0015),
        lng: coords.lng + (((i * 2) % 5 - 2) * 0.0015),
        severity: inc.severity,
        timestamp: new Date(Date.now() - inc.hoursAgo * 3600000).toISOString(),
        status: 'Resolved',
        is_simulated: 0,
        source: 'dataset'
      };
    });
  }

  prepare(sql: string) {
    const s = sql.toLowerCase();

    return {
      run: (params?: any) => {
        if (s.includes('insert into incidents')) {
          const row = {
            id: this.nextIncidentId++,
            type: params?.type || 'Theft',
            location: params?.location || 'Hitech City',
            lat: Number(params?.lat) || 17.4435,
            lng: Number(params?.lng) || 78.3772,
            severity: params?.severity || 'Medium',
            timestamp: params?.timestamp || new Date().toISOString(),
            status: params?.status || 'Active',
            is_simulated: params?.is_simulated ?? 1,
            source: params?.source || 'simulation'
          };
          this.incidents.unshift(row);
          return { lastInsertRowid: row.id, changes: 1 };
        }

        if (s.includes('insert into predictions')) {
          const row = Array.isArray(params)
            ? {
                id: this.nextPredictionId++,
                location_id: params[0],
                hour: params[1],
                density: params[2],
                threat_level: params[3],
                confidence: params[4],
                source: params[5] || 'heuristic fallback',
                timestamp: new Date().toISOString()
              }
            : {
                id: this.nextPredictionId++,
                ...params,
                timestamp: new Date().toISOString()
              };
          this.predictions.unshift(row);
          return { lastInsertRowid: row.id, changes: 1 };
        }

        return { lastInsertRowid: 1, changes: 1 };
      },

      all: () => {
        if (s.includes('from incidents')) {
          if (s.includes('is_simulated = 0')) {
            return this.incidents.filter(i => !i.is_simulated).slice(0, 300);
          }
          if (s.includes('is_simulated = 1')) {
            return this.incidents.filter(i => !!i.is_simulated).slice(0, 100);
          }
          return [...this.incidents].slice(0, 300);
        }
        if (s.includes('from predictions')) {
          return [...this.predictions];
        }
        return [];
      },

      get: () => {
        if (s.includes('count(*)')) {
          return { count: this.incidents.length };
        }
        return null;
      }
    };
  }

  exec(sql: string) {
    // Schema creation is a no-op for in-memory object store
  }

  transaction(fn: any) {
    return (rows: any[]) => {
      for (const row of rows) {
        this.incidents.push({
          id: this.nextIncidentId++,
          ...row
        });
      }
    };
  }
}

// Instantiate Database Engine
let db: any;

try {
  // On Vercel, always use pure in-memory engine to prevent native C++ module failures
  if (process.env.VERCEL) {
    console.log('[Database] Vercel environment detected: using pure in-memory SQLite store');
    db = new InMemoryDatabase();
  } else {
    // In local persistent environment, attempt better-sqlite3 with fallback
    const Database = require('better-sqlite3');
    const dir = path.dirname(dbPath);
    fs.accessSync(dir, fs.constants.W_OK);
    db = new Database(dbPath);
    console.log(`[Database] Connected to persistent SQLite file: ${dbPath}`);
  }
} catch (e: any) {
  console.log(`[Database] Native SQLite unavailable (${e?.message || 'fallback'}), using pure in-memory store`);
  db = new InMemoryDatabase();
}

// Initialize Schema and Seed Data
export function initDB() {
  try {
    if (typeof db.exec === 'function') {
      db.exec(`
        CREATE TABLE IF NOT EXISTS incidents (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          type TEXT NOT NULL,
          location TEXT NOT NULL,
          lat REAL NOT NULL,
          lng REAL NOT NULL,
          severity TEXT NOT NULL,
          timestamp TEXT NOT NULL,
          status TEXT DEFAULT 'Active',
          is_simulated INTEGER DEFAULT 0,
          source TEXT DEFAULT 'dataset'
        );

        CREATE TABLE IF NOT EXISTS training_logs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          model_name TEXT NOT NULL,
          status TEXT NOT NULL,
          timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS predictions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          location_id INTEGER,
          hour INTEGER,
          density INTEGER,
          threat_level INTEGER,
          confidence REAL,
          source TEXT DEFAULT 'heuristic fallback',
          timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        );
      `);
    }

    const count = db.prepare('SELECT count(*) as count FROM incidents').get() as { count: number };
    if (!count || count.count === 0) {
      seedDatabaseFromDataset();
    }
  } catch (err) {
    console.warn('[Database] initDB warning (non-fatal):', err);
  }
}

/**
 * Seed database with real incidents if CSV dataset is available,
 * otherwise baseline records across all 20 Hyderabad sectors are already active.
 */
export function seedDatabaseFromDataset() {
  const dataDir = path.join(__dirname, '../../data');
  const possibleCsvs = [
    path.join(dataDir, 'crime_data.csv'),
    path.join(dataDir, 'hyderabad_crime_data_enhanced (1).csv'),
    path.join(dataDir, 'processed', 'train.csv')
  ];

  let csvPath: string | null = null;
  for (const p of possibleCsvs) {
    if (fs.existsSync(p)) {
      csvPath = p;
      break;
    }
  }

  const insert = db.prepare(`
    INSERT INTO incidents (type, location, lat, lng, severity, timestamp, status, is_simulated, source)
    VALUES (@type, @location, @lat, @lng, @severity, @timestamp, @status, @is_simulated, @source)
  `);

  const insertMany = db.transaction((rows: any[]) => {
    for (const row of rows) insert.run(row);
  });

  if (csvPath) {
    console.log(`[Database] Seeding SQLite from real dataset CSV: ${csvPath}`);
    try {
      const content = fs.readFileSync(csvPath, 'utf8');
      const lines = content.split('\n').filter(l => l.trim().length > 0);
      if (lines.length > 1) {
        const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
        const areaIdx = headers.findIndex(h => ['Area', 'Location', 'District'].includes(h));
        const typeIdx = headers.findIndex(h => ['Crime_Type', 'type', 'Category'].includes(h));
        const sevIdx = headers.findIndex(h => ['Severity', 'Severity_Level'].includes(h));
        const timeIdx = headers.findIndex(h => ['timestamp', 'Timestamp', 'Date', 'Time'].includes(h));

        const rows: any[] = [];
        for (let i = 1; i < Math.min(lines.length, 300); i++) {
          const parts = lines[i].split(',').map(p => p.trim().replace(/"/g, ''));
          const area = areaIdx !== -1 && parts[areaIdx] ? parts[areaIdx] : "Hitech City";
          const coords = LOCATIONS_COORDS[area] || LOCATIONS_COORDS["Hitech City"];
          const type = typeIdx !== -1 && parts[typeIdx] ? parts[typeIdx] : "Theft";
          const severity = sevIdx !== -1 && parts[sevIdx] ? parts[sevIdx] : "Medium";
          const ts = timeIdx !== -1 && parts[timeIdx] ? parts[timeIdx] : new Date().toISOString();

          rows.push({
            type,
            location: area,
            lat: coords.lat + ((i % 10 - 5) * 0.001),
            lng: coords.lng + (((i * 3) % 10 - 5) * 0.001),
            severity: severity === '0' ? 'Low' : severity === '1' ? 'Medium' : severity === '2' ? 'High' : severity,
            timestamp: ts.includes('T') ? ts : new Date(ts).toISOString(),
            status: 'Resolved',
            is_simulated: 0,
            source: 'dataset'
          });
        }
        insertMany(rows);
        console.log(`[Database] Successfully seeded ${rows.length} real historical incidents from dataset.`);
        return;
      }
    } catch (e) {
      console.error("[Database] Error reading CSV during seed, using baseline seed records:", e);
    }
  }

  // Populate all 20 Hyderabad sectors
  const initialRows = BASELINE_INCIDENTS.map((inc, i) => {
    const coords = LOCATIONS_COORDS[inc.location] || { lat: 17.3850, lng: 78.4867 };
    return {
      type: inc.type,
      location: inc.location,
      lat: coords.lat + ((i % 5 - 2) * 0.0015),
      lng: coords.lng + (((i * 2) % 5 - 2) * 0.0015),
      severity: inc.severity,
      timestamp: new Date(Date.now() - inc.hoursAgo * 3600000).toISOString(),
      status: 'Resolved',
      is_simulated: 0,
      source: 'dataset'
    };
  });

  insertMany(initialRows);
  console.log(`[Database] Seeded ${initialRows.length} historical baseline incidents covering all 20 sectors.`);
}

export default db;
