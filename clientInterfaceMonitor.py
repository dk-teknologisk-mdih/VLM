import socket
import struct
import sys
import time

ROBOT_HOST = "127.0.0.1"   # <-- change to your robot IP
ROBOT_PORT = 30001            # Primary interface

MESSAGE_TYPE_ROBOT_MESSAGE = 20

ROBOT_MESSAGE_TYPE_TEXT = 0
ROBOT_MESSAGE_TYPE_POPUP = 2
ROBOT_MESSAGE_TYPE_SAFETY_MODE = 5
ROBOT_MESSAGE_TYPE_RUNTIME_EXCEPTION = 10


def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed by peer")
        data += chunk
    return data


def read_message(sock):
    header = recv_exact(sock, 4)
    message_size = struct.unpack(">i", header)[0]
    body = recv_exact(sock, message_size - 4)
    return header + body


def read_cstring_or_rest(data, offset):
    if offset >= len(data):
        return "", offset
    raw = data[offset:]
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        text = repr(raw)
    return text, len(data)


def read_uint8(data, offset):
    return data[offset], offset + 1


def read_bool(data, offset):
    return bool(data[offset]), offset + 1


def read_int32(data, offset):
    value = struct.unpack_from(">i", data, offset)[0]
    return value, offset + 4


def read_uint32(data, offset):
    value = struct.unpack_from(">I", data, offset)[0]
    return value, offset + 4


def read_uint64(data, offset):
    value = struct.unpack_from(">Q", data, offset)[0]
    return value, offset + 8


def parse_robot_message(msg):
    offset = 0

    message_size, offset = read_int32(msg, offset)
    message_type, offset = read_uint8(msg, offset)

    if message_type != MESSAGE_TYPE_ROBOT_MESSAGE:
        return None

    timestamp, offset = read_uint64(msg, offset)
    source, offset = read_uint8(msg, offset)
    robot_message_type, offset = read_uint8(msg, offset)

    result = {
        "message_size": message_size,
        "timestamp": timestamp,
        "source": source,
        "robot_message_type": robot_message_type,
    }

    if robot_message_type == ROBOT_MESSAGE_TYPE_SAFETY_MODE:
        robot_message_code, offset = read_int32(msg, offset)
        robot_message_argument, offset = read_int32(msg, offset)
        safety_mode_type, offset = read_uint8(msg, offset)
        report_data_type, offset = read_uint32(msg, offset)
        report_data, offset = read_uint32(msg, offset)

        result.update({
            "kind": "SafetyModeMessage",
            "robot_message_code": robot_message_code,
            "robot_message_argument": robot_message_argument,
            "safety_mode_type": safety_mode_type,
            "report_data_type": report_data_type,
            "report_data": report_data,
        })
        return result

    elif robot_message_type == ROBOT_MESSAGE_TYPE_POPUP:
        request_id, offset = read_uint32(msg, offset)
        requested_type, offset = read_uint32(msg, offset)
        warning, offset = read_bool(msg, offset)
        error, offset = read_bool(msg, offset)
        blocking, offset = read_bool(msg, offset)
        title_size, offset = read_uint8(msg, offset)

        title_bytes = msg[offset:offset + title_size]
        popup_title = title_bytes.decode("utf-8", errors="replace")
        offset += title_size

        popup_text, offset = read_cstring_or_rest(msg, offset)

        result.update({
            "kind": "PopupMessage",
            "request_id": request_id,
            "requested_type": requested_type,
            "warning": warning,
            "error": error,
            "blocking": blocking,
            "popup_title": popup_title,
            "popup_text": popup_text,
        })
        return result

    elif robot_message_type == ROBOT_MESSAGE_TYPE_TEXT:
        text, offset = read_cstring_or_rest(msg, offset)
        result.update({
            "kind": "TextMessage",
            "text": text,
        })
        return result

    elif robot_message_type == ROBOT_MESSAGE_TYPE_RUNTIME_EXCEPTION:
        script_line_number, offset = read_int32(msg, offset)
        script_column_number, offset = read_int32(msg, offset)
        exception_text, offset = read_cstring_or_rest(msg, offset)

        result.update({
            "kind": "RuntimeExceptionMessage",
            "script_line_number": script_line_number,
            "script_column_number": script_column_number,
            "exception_text": exception_text,
        })
        return result

    return None


def main():
    print(f"Connecting to {ROBOT_HOST}:{ROBOT_PORT} ...")
    with socket.create_connection((ROBOT_HOST, ROBOT_PORT), timeout=5) as sock:
        sock.settimeout(None)
        print("Connected. Monitoring primary interface...")

        while True:
            try:
                msg = read_message(sock)
                if len(msg) < 5:
                    continue

                message_type = msg[4]

                if message_type != MESSAGE_TYPE_ROBOT_MESSAGE:
                    continue

                parsed = parse_robot_message(msg)
                if not parsed:
                    continue

                kind = parsed["kind"]
                ts = parsed["timestamp"]
                src = parsed["source"]

                if kind == "SafetyModeMessage":
                    print(
                        f"[SafetyModeMessage] ts={ts} src={src} "
                        f"code={parsed['robot_message_code']} "
                        f"arg={parsed['robot_message_argument']} "
                        f"safety_mode_type={parsed['safety_mode_type']} "
                        f"report_data_type={parsed['report_data_type']} "
                        f"report_data={parsed['report_data']}"
                    )

                elif kind == "PopupMessage":
                    print(
                        f"[PopupMessage] ts={ts} src={src} "
                        f"request_id={parsed['request_id']} "
                        f"requested_type={parsed['requested_type']} "
                        f"warning={parsed['warning']} "
                        f"error={parsed['error']} "
                        f"blocking={parsed['blocking']} "
                        f"title={parsed['popup_title']!r} "
                        f"text={parsed['popup_text']!r}"
                    )

                elif kind == "TextMessage":
                    print(
                        f"[TextMessage] ts={ts} src={src} "
                        f"text={parsed['text']!r}"
                    )

                elif kind == "RuntimeExceptionMessage":
                    print(
                        f"[RuntimeExceptionMessage] ts={ts} src={src} "
                        f"line={parsed['script_line_number']} "
                        f"column={parsed['script_column_number']} "
                        f"text={parsed['exception_text']!r}"
                    )

            except KeyboardInterrupt:
                print("\nStopped by user.")
                break
            except Exception as e:
                print(f"Connection/parser error: {e}", file=sys.stderr)
                break


if __name__ == "__main__":
    main()