"""Offline tests for path-inspection waypoint verification and VLM ordering."""

from copy import deepcopy
from io import StringIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import yaml
from PIL import Image

from vlm_programming_orchestrator import orchestrator as orchestrator_module
from vlm_programming_orchestrator.config import Config, State
from vlm_programming_orchestrator.tasks.path_inspection import backend, pipeline
from vlm_programming_orchestrator.tasks.path_inspection.pipeline import (
    WaypointOrderingError,
    WaypointVerificationError,
    order_inspection_waypoints,
    verify_inspection_waypoints,
)


class WaypointVerificationTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (80, 60))
        self.initial = [{"point": [100, 200], "label": "joint"}]
        self.corrected = [{"point": [120, 220], "label": "corrected joint"}]
        self.detector = Mock()
        self.criteria = "Inspect the test frame; include all joints."

    def response(self, accepted, waypoints=None):
        return {
            "accepted": accepted,
            "reason": "Reviewed against the image",
            "waypoints": deepcopy(self.initial if waypoints is None else waypoints),
        }

    def verify(self, attempts=3):
        return verify_inspection_waypoints(
            self.detector, self.image, self.initial, self.criteria, attempts)

    def test_immediate_acceptance(self):
        self.detector.detect_objects.return_value = self.response(True)
        self.assertEqual(self.verify(), self.initial)
        self.detector.detect_objects.assert_called_once()
        call = self.detector.detect_objects.call_args
        self.assertIs(call.args[0], self.image)
        self.assertIn(self.criteria, call.args[1])
        self.assertIn(yaml.safe_dump(self.initial, sort_keys=False), call.args[1])

    def test_corrections_are_submitted_for_acceptance(self):
        self.detector.detect_objects.side_effect = [
            self.response(False, self.corrected), self.response(True, self.corrected),
        ]
        self.assertEqual(self.verify(), self.corrected)
        calls = self.detector.detect_objects.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertIn(yaml.safe_dump(self.corrected, sort_keys=False), calls[1].args[1])
        self.assertTrue(all(call.args[0] is self.image for call in calls))
        self.assertEqual(self.initial[0]["point"], [100, 200])

    def test_overlay_updates_after_each_review_including_invalid_responses(self):
        rendered = []
        self.detector.draw_points_on_image.side_effect = (
            lambda image, points, output_path: rendered.append(
                (image, deepcopy(points), output_path)))
        self.detector.detect_objects.side_effect = [
            self.response(False, self.corrected), [], self.response(True, self.corrected),
        ]
        self.assertEqual(self.verify(), self.corrected)
        self.assertEqual(len(rendered), 7)
        self.assertEqual(rendered[0][1:], (self.initial, "1_inspection_path.png"))
        for attempt in range(1, 4):
            snapshot, latest = rendered[2 * attempt - 1:2 * attempt + 1]
            self.assertEqual(snapshot[1:], (
                self.corrected, f"1_inspection_path_verification_{attempt}.png"))
            self.assertEqual(latest[1:], (self.corrected, "1_inspection_path.png"))
        self.assertTrue(all(render[0] is self.image for render in rendered))

    def test_acceptance_on_last_attempt(self):
        self.detector.detect_objects.side_effect = [
            self.response(False), self.response(False), self.response(True),
        ]
        self.assertEqual(self.verify(), self.initial)
        self.assertEqual(self.detector.detect_objects.call_count, 3)

    def test_exhaustion_is_bounded(self):
        self.detector.detect_objects.return_value = self.response(False)
        with self.assertRaisesRegex(WaypointVerificationError, "after 3"):
            self.verify()
        self.assertEqual(self.detector.detect_objects.call_count, 3)

    def test_final_correction_is_not_implicitly_accepted(self):
        self.detector.detect_objects.return_value = self.response(False, self.corrected)
        with self.assertRaises(WaypointVerificationError):
            self.verify(attempts=1)

    def test_changed_points_marked_accepted_require_another_review(self):
        self.detector.detect_objects.side_effect = [
            self.response(True, self.corrected), self.response(True, self.corrected),
        ]
        self.assertEqual(self.verify(), self.corrected)
        self.assertEqual(self.detector.detect_objects.call_count, 2)

    def test_invalid_responses_consume_attempt_without_replacing_waypoints(self):
        invalid_responses = [
            [], None, "accepted", {}, {"accepted": "false"},
            {"accepted": True, "reason": "ok", "waypoints": []},
            {"accepted": True, "waypoints": self.initial},
            {"accepted": True, "reason": 123, "waypoints": self.initial},
        ]
        for invalid in invalid_responses:
            with self.subTest(response=invalid):
                self.detector.reset_mock()
                self.detector.detect_objects.side_effect = [invalid, self.response(True)]
                self.assertEqual(self.verify(attempts=2), self.initial)
                retry_prompt = self.detector.detect_objects.call_args.args[1]
                self.assertIn("Invalid verification response", retry_prompt)
                self.assertIn(yaml.safe_dump(self.initial, sort_keys=False), retry_prompt)

    def test_invalid_waypoints_cannot_be_accepted(self):
        invalid_waypoints = [
            [], {}, [None], [{"point": [1, 2], "label": ""}],
            [{"point": [1, 2, 3], "label": "joint"}],
        ]
        for coordinate in (-1, 1001, float("nan"), float("inf"), True, "100"):
            invalid_waypoints.append([{"point": [coordinate, 200], "label": "joint"}])
        for waypoints in invalid_waypoints:
            with self.subTest(waypoints=waypoints):
                self.detector.detect_objects.return_value = self.response(True, waypoints)
                with self.assertRaises(WaypointVerificationError):
                    self.verify(attempts=1)

    def test_invalid_initial_points_fail_before_call(self):
        with self.assertRaisesRegex(WaypointVerificationError, "Invalid detected"):
            verify_inspection_waypoints(self.detector, self.image, [], self.criteria)
        self.detector.detect_objects.assert_not_called()

    def test_invalid_limits_fail_before_call(self):
        for attempts in (0, -1, True, 1.5, "3"):
            with self.subTest(attempts=attempts):
                with self.assertRaises(ValueError):
                    self.verify(attempts)
        self.detector.detect_objects.assert_not_called()


class WaypointOrderingTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (80, 60))
        self.waypoints = [
            {"point": [500, 500], "label": "joint"},
            {"point": [100, 500], "label": "endpoint: upper tube"},
            {"point": [500, 100], "label": "endpoint: left tube"},
            {"point": [500, 900], "label": "endpoint: right tube"},
        ]
        self.response = {
            "routable": True,
            "reason": "Retrace one branch and finish at another endpoint.",
            "tube_segments": [[1, 2], [1, 3], [1, 4]],
            "waypoint_order": [2, 1, 3, 1, 4],
        }
        self.detector = Mock()
        self.detector.detect_objects.return_value = self.response

    def order(self, waypoints=None):
        return order_inspection_waypoints(
            self.detector, self.image,
            self.waypoints if waypoints is None else waypoints,
            "Inspect the test frame.")

    def test_vlm_order_is_used_verbatim_with_retracing_and_open_finish(self):
        original = deepcopy(self.waypoints)
        result = self.order()
        self.assertEqual(result, [original[index - 1] for index in [2, 1, 3, 1, 4]])
        self.assertNotEqual(result[0], result[-1])
        self.assertEqual(self.waypoints, original)
        self.detector.detect_objects.assert_called_once()
        call = self.detector.detect_objects.call_args
        self.assertIs(call.args[0], self.image)
        self.assertIn("Inspect the test frame.", call.args[1])
        indexed = [{"id": index, **point} for index, point in enumerate(original, start=1)]
        self.assertIn(yaml.safe_dump(indexed, sort_keys=False), call.args[1])
        self.assertEqual(call.kwargs["save_path"], "1_inspection_path_ordering.yaml")

    def test_repeated_visits_are_independent_for_depth_conversion(self):
        result = self.order()
        self.assertIsNot(result[1], result[3])
        self.assertIsNot(result[1]["point"], result[3]["point"])
        pipeline.add_depth_to_detections(result, None, self.image.size, hardcoded=True)
        self.assertTrue(all(len(point["point"]) == 3 for point in result))
        self.assertTrue(all(len(point["point"]) == 2 for point in self.waypoints))

    def test_single_point_needs_no_segments(self):
        self.response.update(tube_segments=[], waypoint_order=[1])
        self.assertEqual(self.order(self.waypoints[:1]), self.waypoints[:1])

    def test_closed_route_is_allowed(self):
        self.response["waypoint_order"] = [1, 2, 1, 3, 1, 4, 1]
        self.assertEqual(len(self.order()), 7)

    def test_invalid_responses_fail_without_retry(self):
        invalid = [None, [], "route", {}]
        invalid.extend({key: value for key, value in self.response.items() if key != missing}
                       for missing in self.response)
        updates = [
            {"routable": "true"}, {"routable": 1}, {"reason": " "}, {"reason": 1},
            {"routable": False, "reason": "Disconnected frame"},
            {"waypoint_order": []}, {"waypoint_order": "2,1,3,1,4"},
            {"waypoint_order": [2, 1, 3, 4]},
            {"waypoint_order": [2, 1, 3, 1, 1, 4]},
            {"waypoint_order": [2, 1, 3, 1]},
            {"tube_segments": None}, {"tube_segments": []},
            {"tube_segments": [[1, 2], [1, 3], [1, 4], [2, 3]]},
            {"tube_segments": [[1, 2], [1, 3], [1, 4], [2, 1]]},
            {"tube_segments": [[1, 2], [1, 3], [1, 4], [1, 1]]},
            {"tube_segments": [[1, 2], [1, 3], [1, 4], [1]]},
            {"tube_segments": [[1, 2], [1, 3], [1, 4], "1,2"]},
        ]
        for bad_id in (True, 1.0, "1", 0, -1, 5, None, []):
            updates.append({"waypoint_order": [2, bad_id, 3, 1, 4]})
            updates.append({"tube_segments": [[bad_id, 2], [1, 3], [1, 4]]})
        invalid.extend({**self.response, **update} for update in updates)
        for response in invalid:
            with self.subTest(response=response):
                self.detector.reset_mock()
                self.detector.detect_objects.return_value = response
                with self.assertRaises(WaypointOrderingError):
                    self.order()
                self.detector.detect_objects.assert_called_once()

    def test_invalid_input_fails_before_call(self):
        with self.assertRaises(WaypointOrderingError):
            self.order([])
        self.detector.detect_objects.assert_not_called()


class PathInspectionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        original_cwd = os.getcwd()
        self.addCleanup(os.chdir, original_cwd)
        os.chdir(self.temp_dir.name)
        self.image = Image.new("RGB", (80, 60))
        self.initial = [{"point": [100, 200], "label": "joint"}]
        self.corrected = [{"point": [120, 220], "label": "corrected joint"}]
        self.detector = Mock()
        self.detector.capture_realsense_image.return_value = (self.image, None, None)
        self.factory = self.enterContext(
            patch.object(pipeline, "get_detector_backend", return_value=self.detector))
        self.convert = self.enterContext(
            patch.object(pipeline, "convert_plan_to_3d", return_value=[{"step": 1}]))

    def response(self, accepted, waypoints):
        return {"accepted": accepted, "reason": "reviewed", "waypoints": deepcopy(waypoints)}

    def ordering_response(self, order=None, segments=None):
        return {
            "routable": True,
            "reason": "Continuous tube traversal",
            "tube_segments": [] if segments is None else segments,
            "waypoint_order": [1] if order is None else order,
        }

    def test_only_approved_points_are_saved_and_converted(self):
        self.detector.detect_objects.side_effect = [
            deepcopy(self.initial), self.response(False, self.corrected),
            self.response(True, self.corrected), self.ordering_response(),
        ]
        with patch.object(pipeline, "add_depth_to_detections",
                          wraps=pipeline.add_depth_to_detections) as add_depth:
            result = pipeline.plan_inspection_path("test-key", "test-url")
        self.assertEqual(self.detector.detect_objects.call_count, 4)
        add_depth.assert_called_once()
        self.convert.assert_called_once()
        expected_depth_points = deepcopy(self.corrected)
        expected_depth_points[0]["point"].append(990)
        self.assertEqual(result["detections"], expected_depth_points)
        saved = yaml.safe_load(Path("VLM_output/1_inspection_path_pixel.yaml").read_text())
        self.assertEqual(saved, self.corrected)
        self.assertEqual(self.convert.call_args.args[1][0]["trajectory"], expected_depth_points)
        self.assertEqual(self.detector.draw_points_on_image.call_count, 6)
        self.detector.draw_points_on_image.assert_called_with(
            self.image, expected_depth_points, output_path="1_inspection_path.png")
        calls = self.detector.detect_objects.call_args_list
        self.assertEqual(calls[0].kwargs["save_path"], "1_inspection_path_initial.yaml")
        self.assertEqual(calls[2].kwargs["save_path"], "1_inspection_path_verification_2.yaml")
        self.assertEqual(calls[3].kwargs["save_path"], "1_inspection_path_ordering.yaml")
        self.assertTrue(all(call.args[0] is self.image for call in calls))
        self.assertIn(yaml.safe_dump(
            [{"id": 1, **self.corrected[0]}], sort_keys=False), calls[3].args[1])

    def test_ordered_repeated_visits_are_saved_and_converted(self):
        accepted = [
            {"point": [500, 500], "label": "joint"},
            {"point": [100, 500], "label": "upper endpoint"},
            {"point": [500, 100], "label": "left endpoint"},
            {"point": [500, 900], "label": "right endpoint"},
        ]
        order = [2, 1, 3, 1, 4]
        self.detector.detect_objects.side_effect = [
            deepcopy(accepted), self.response(True, accepted),
            self.ordering_response(order, [[1, 2], [1, 3], [1, 4]]),
        ]
        result = pipeline.plan_inspection_path("test-key", "test-url")
        expected = [deepcopy(accepted[point_id - 1]) for point_id in order]
        saved = yaml.safe_load(Path("VLM_output/1_inspection_path_pixel.yaml").read_text())
        self.assertEqual(saved, expected)
        for waypoint in expected:
            waypoint["point"].append(990)
        self.assertEqual(result["detections"], expected)
        self.assertEqual(self.convert.call_args.args[1][0]["trajectory"], expected)
        self.assertEqual(self.detector.detect_objects.call_count, 3)

    def test_ordering_failure_stops_before_depth_save_and_conversion(self):
        self.detector.detect_objects.side_effect = [
            deepcopy(self.initial), self.response(True, self.initial),
            {"routable": False, "reason": "Disconnected tubes",
             "tube_segments": [], "waypoint_order": []},
        ]
        with patch.object(pipeline, "add_depth_to_detections") as add_depth:
            with self.assertRaisesRegex(WaypointOrderingError, "Disconnected tubes"):
                pipeline.plan_inspection_path("test-key", "test-url")
        add_depth.assert_not_called()
        self.convert.assert_not_called()
        self.assertEqual(self.detector.detect_objects.call_count, 3)
        self.assertFalse(Path("VLM_output/1_inspection_path_pixel.yaml").exists())

    def test_no_depth_or_conversion_on_verification_failure(self):
        self.detector.detect_objects.side_effect = [
            deepcopy(self.initial), self.response(False, self.corrected),
        ]
        with patch.object(pipeline, "add_depth_to_detections") as add_depth:
            with self.assertRaises(WaypointVerificationError):
                pipeline.plan_inspection_path("test-key", "test-url",
                                              vlm_verification_max_attempts=1)
        add_depth.assert_not_called()
        self.convert.assert_not_called()
        self.assertEqual(self.detector.draw_points_on_image.call_count, 3)
        self.detector.draw_points_on_image.assert_called_with(
            self.image, self.corrected, output_path="1_inspection_path.png")
        self.assertFalse(Path("VLM_output/1_inspection_path_pixel.yaml").exists())

    def test_custom_prompt_and_input_image_are_reused(self):
        prompt_path = Path("custom.txt")
        prompt_path.write_text("Inspect {target_object}; custom criteria.", encoding="utf-8")
        self.image.save("input.png")
        self.detector.detect_objects.side_effect = [
            deepcopy(self.initial), self.response(True, self.initial),
            self.ordering_response(),
        ]
        pipeline.plan_inspection_path(
            "test-key", "test-url", target_object="test frame",
            vlm_prompt_file=prompt_path, image_path="input.png")
        calls = self.detector.detect_objects.call_args_list
        self.assertIs(calls[0].args[0], calls[1].args[0])
        self.assertIn("Inspect test frame; custom criteria.", calls[1].args[1])
        self.assertIs(calls[0].args[0], calls[2].args[0])
        self.assertIn("Inspect test frame; custom criteria.", calls[2].args[1])
        self.detector.capture_realsense_image.assert_not_called()
        self.assertFalse(self.factory.call_args.kwargs["use_realsense"])

    def test_no_detection_returns_none_without_review(self):
        self.detector.detect_objects.return_value = []
        self.assertIsNone(pipeline.plan_inspection_path("test-key", "test-url"))
        self.detector.detect_objects.assert_called_once()
        self.convert.assert_not_called()

    def test_invalid_limit_fails_before_camera_initialization(self):
        with self.assertRaises(ValueError):
            pipeline.plan_inspection_path("test-key", "test-url",
                                          vlm_verification_max_attempts=0)
        self.factory.assert_not_called()

    def test_backend_forwards_configured_limit(self):
        config = Config(vlm_verification_max_attempts=5, display_name=None)
        with patch.object(backend, "plan_inspection_path") as plan:
            backend.PathInspectionTaskBackend(config).plan("key", "url", {}, "test-backend")
        self.assertEqual(plan.call_args.kwargs["vlm_verification_max_attempts"], 5)

    def test_config_default_and_invalid_limits(self):
        self.assertEqual(Config(display_name=None).vlm_verification_max_attempts, 3)
        for attempts in (0, -1, True, 1.5, "3"):
            with self.subTest(attempts=attempts):
                with self.assertRaises(ValueError):
                    Config(vlm_verification_max_attempts=attempts, display_name=None)

    def make_orchestrator(self, **overrides):
        config = Config(
            api_key="test-key", base_url="test-url", task_type="path_inspection",
            local_output_dir=Path("generated"), ftp_shared_dir=Path("ftp"),
            display_name=None, **overrides)
        for factory in ("get_robot_backend", "get_codegen_backend", "get_task_backend"):
            self.enterContext(patch.object(orchestrator_module, factory, return_value=Mock()))
        return orchestrator_module.VLMOrchestrator(config, gui_config={})

    def test_verification_failure_stops_before_codegen(self):
        orchestrator = self.make_orchestrator()
        orchestrator.task_backend.plan.side_effect = WaypointVerificationError("review limit reached")
        with patch.object(orchestrator, "_state_generate_code") as generate_code:
            orchestrator.run()
        self.assertEqual(orchestrator.state, State.ERROR)
        self.assertEqual(orchestrator.error_message, "review limit reached")
        self.assertIsNone(orchestrator.plan_data)
        generate_code.assert_not_called()
        orchestrator.codegen_backend.generate_code.assert_not_called()
        orchestrator.robot_backend.assert_not_called()

    def test_saved_plan_still_bypasses_vlm(self):
        plan_path = Path("saved.yaml")
        plan_path.write_text(yaml.safe_dump(self.initial), encoding="utf-8")
        orchestrator = self.make_orchestrator(vlm_plan_file=plan_path)
        orchestrator._state_plan_task()
        self.assertEqual(orchestrator.state, State.GENERATE_CODE)
        orchestrator.task_backend.plan.assert_not_called()

    def test_ordering_failure_stops_before_codegen(self):
        orchestrator = self.make_orchestrator()
        orchestrator.task_backend.plan.side_effect = WaypointOrderingError("Disconnected tubes")
        with patch.object(orchestrator, "_state_generate_code") as generate_code:
            orchestrator.run()
        self.assertEqual(orchestrator.state, State.ERROR)
        self.assertEqual(orchestrator.error_message, "Disconnected tubes")
        self.assertIsNone(orchestrator.plan_data)
        generate_code.assert_not_called()
        orchestrator.codegen_backend.generate_code.assert_not_called()
        orchestrator.robot_backend.assert_not_called()

    def test_codegen_prompt_preserves_repeated_visits(self):
        config = Config(display_name=None)
        task = backend.PathInspectionTaskBackend(config)
        trajectory = [{"step": 1, "action": "inspect", "trajectory": self.initial * 2}]
        original_cwd = os.getcwd()
        try:
            os.chdir(Path(__file__).parent)
            prompt = task.build_prompt(trajectory, "UR", "URScript", "task.txt", "rules.txt")
        finally:
            os.chdir(original_cwd)
        self.assertIn("EVERY repeated waypoint visit", prompt)
        self.assertIn("do not add a closing inspection move", prompt)
        self.assertIn(yaml.dump(trajectory, default_flow_style=False, sort_keys=False), prompt)

    def test_saved_code_still_bypasses_planning(self):
        orchestrator = self.make_orchestrator(code_file=Path("saved.script"))
        self.assertEqual(orchestrator.state, State.GENERATE_CODE)
        orchestrator.task_backend.plan.assert_not_called()

    def test_real_detector_parses_and_saves_verification_mapping(self):
        from vlm_programming_orchestrator.vision.gemini_realsense.detector import (
            GeminiRoboticsDetector,
        )

        detector = GeminiRoboticsDetector(
            "test-key", base_url="https://example.invalid", use_realsense=False)
        review = self.response(True, self.initial)
        http_response = Mock(status_code=200)
        http_response.json.return_value = {
            "candidates": [{"content": {"parts": [
                {"text": "```yaml\n" + yaml.safe_dump(review) + "```"},
            ]}}],
        }
        with patch(
            "vlm_programming_orchestrator.vision.gemini_realsense.detector.requests.post",
            return_value=http_response,
        ) as post:
            result = verify_inspection_waypoints(
                detector, self.image, self.initial, "Inspect test frame")
        self.assertEqual(result, self.initial)
        post.assert_called_once()
        parts = post.call_args.kwargs["json"]["contents"][0]["parts"]
        self.assertEqual(parts[0]["inlineData"]["mimeType"], "image/jpeg")
        self.assertTrue(parts[0]["inlineData"]["data"])
        self.assertIn("Inspect test frame", parts[1]["text"])
        for filename in ("1_inspection_path.png", "1_inspection_path_verification_1.png"):
            with Image.open(Path("VLM_output") / filename) as overlay:
                self.assertEqual(overlay.size, self.image.size)
                self.assertIsNotNone(overlay.getbbox())
        self.assertIsNone(self.image.getbbox())
        saved = yaml.safe_load(
            Path("VLM_output/1_inspection_path_verification_1.yaml").read_text())
        self.assertEqual(saved, review)

    def test_cli_forwards_verification_limit(self):
        import main as entry_point

        with (
            patch.object(entry_point, "load_dotenv"),
            patch.dict(os.environ, {"API_KEY": "test-key", "BASE_URL": "https://example.invalid"}),
            patch("sys.argv", ["main.py", "--task-type", "path_inspection",
                               "--vlm-verification-max-attempts", "5"]),
            patch.object(entry_point, "VLMOrchestrator") as factory,
        ):
            entry_point.main()
        self.assertEqual(factory.call_args.args[0].vlm_verification_max_attempts, 5)
        factory.return_value.run.assert_called_once()
        factory.return_value.shutdown.assert_called_once()

    def test_cli_rejects_invalid_limits_before_pipeline_start(self):
        import main as entry_point

        for limit in ("0", "-1", "invalid"):
            with (
                self.subTest(limit=limit),
                patch.object(entry_point, "load_dotenv"),
                patch("sys.argv", ["main.py", "--vlm-verification-max-attempts", limit]),
                patch("sys.stderr", new_callable=StringIO),
                patch.object(entry_point, "VLMOrchestrator") as factory,
            ):
                with self.assertRaises(SystemExit) as error:
                    entry_point.main()
                self.assertEqual(error.exception.code, 2)
                factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()