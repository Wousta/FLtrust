#!/bin/bash

srvr_ip=127.0.0.1
port=2000

# Function to clean up only the srvr and clnt processes
cleanup() {
  echo "Terminating srvr and clnt processes..."
  kill $SRVR_PID $CLNT_PID 2>/dev/null
  exit 0
}

# Trap SIGINT and SIGTERM to run cleanup.
trap cleanup SIGINT SIGTERM

# Launch redis (disowned so it is not affected)
echo "Starting Redis server on $srvr_ip:$port"
redis-server --bind "$srvr_ip" --port "$port" >/dev/null &
disown
sleep 1
redis-cli -h "$srvr_ip" -p "$port" SET srvr "0" >/dev/null
redis-cli -h "$srvr_ip" -p "$port" SET clnt "0" >/dev/null
redis-cli -h "$srvr_ip" -p "$port" SET nid "0" >/dev/null

echo "Redis server started on $srvr_ip:$port"

rm -rf logs/*

# Start the srvr process and capture its PID.
build/srvr --srvr_ip $srvr_ip --port $port &
SRVR_PID=$!

# Start the clnt process and capture its PID.
build/clnt --srvr_ip $srvr_ip --port $port &
CLNT_PID=$!

# Wait for these processes.
wait $SRVR_PID $CLNT_PID
