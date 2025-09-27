import socket
import sys
import io
import os
import signal
import argparse

# A special token to signal the end of a message.
EOM_TOKEN = "<|EOM|>"

# This is now our session manager.
sessions = {}

# Custom exception for timeouts
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    """Handler to raise an exception when the alarm signal is received."""
    raise TimeoutException("Execution timed out")

def execute_code(session_id, code, timeout_seconds):
    """Executes code within a specific session's state with a timeout."""
    if session_id not in sessions:
        sessions[session_id] = {}
    
    session_state = sessions[session_id]
    workspace_dir = os.path.join('/app/workspaces', session_id)
    
    original_dir = os.getcwd()
    os.chdir(workspace_dir)

    # Set the signal handler for the alarm signal
    signal.signal(signal.SIGALRM, timeout_handler)
    # Set the alarm
    signal.alarm(timeout_seconds)

    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    try:
        exec(code, session_state)
        output = redirected_output.getvalue()
        # If execution completes successfully, cancel the alarm
        signal.alarm(0)
        return output
    except TimeoutException as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"
    finally:
        # Always ensure the alarm is cancelled and stdout is restored
        signal.alarm(0)
        sys.stdout = old_stdout
        os.chdir(original_dir)


def main():
    """Listens for connections and executes code in sandboxed sessions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=60, help="Timeout for code execution in seconds.")
    args = parser.parse_args()
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', 9999))
    server_socket.listen(5)
    print(f"Python session server is listening on port 9999 with a {args.timeout}s timeout...")

    while True:
        conn, addr = server_socket.accept()
        print(f"Accepted connection from {addr}")
        try:
            full_message = ""
            while True:
                data = conn.recv(4096).decode('utf-8')
                if not data:
                    break
                full_message += data
                
                # Process only when the EOM token is received (for multi-chunk messages)
                if EOM_TOKEN in full_message:
                    # For now, we assume one message per connection, but this is more robust
                    message = full_message.replace(EOM_TOKEN, "")
                    parts = message.split('|', 1)
                    if len(parts) != 2:
                        conn.sendall(b"Error: Invalid message format. Expected 'session_id|code'.")
                        continue

                    session_id, code = parts

                    print(f"=== Session {session_id} ===")
                    print("Executing Python code:")
                    lines = code.split('\n')
                    for i, line in enumerate(lines, 1):
                        print(f"{i:3d} | {line}")
                    print("-" * 30)

                    result = execute_code(session_id, code, args.timeout)

                    if not result.strip():
                        result = "Success: Code executed with no output."

                    print(f"Result: {result}")
                    print("=" * 30)

                    message_to_send = result + EOM_TOKEN
                    conn.sendall(message_to_send.encode('utf-8'))
        except socket.error as e:
            print(f"Socket error: {e}")
        finally:
            print(f"Closing connection from {addr}")
            conn.close()

if __name__ == "__main__":
    main()