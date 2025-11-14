using UnityEngine;
using static UnityEngine.Object;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using Unity.MLAgents.Policies;
using Newtonsoft.Json;
using System.Collections.Generic;

public static class UnityBootstrap
{
    private static EnvSideChannel _channel;
    private static GameObject _go;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    static void Init()
    {
        if (_go != null) return;
        _go = new GameObject("MLAgentsBootstrap(Auto)");
        Object.DontDestroyOnLoad(_go);
        _go.AddComponent<Registrar>();
    }

    private class Registrar : MonoBehaviour
    {
        void Awake()
        {
            _channel = new EnvSideChannel();
            Application.logMessageReceived += _channel.SendDebugStatementToPython;
            SideChannelManager.RegisterSideChannel(_channel);
            Debug.Log("EnvSideChannel registered (auto-bootstrap).");
            SendBehaviorData();
        }

        void SendBehaviorData()
        {
            var behavior = FindFirstObjectByType<BehaviorParameters>(FindObjectsInactive.Include);

            if (behavior == null)
            {
                UnityEngine.Debug.Log("Behavior is Null");
            }

            var features = new Dictionary<string, object>
            {
                ["model"] = "",
                ["inference_device"] = behavior.InferenceDevice.ToString(),
                ["deterministic_inference"] = behavior.DeterministicInference,
                ["behavior_type"] = behavior.BehaviorType.ToString(),
                ["team_id"] = behavior.TeamId,
                ["use_child_actuators"] = behavior.UseChildActuators,
                ["use_child_sensors"] = behavior.UseChildSensors,
                ["observational_attribute_handling"] = behavior.ObservableAttributeHandling.ToString(),
                ["max_step"] = behavior.GetComponent<Agent>().MaxStep,
                ["decision_period"] = behavior.GetComponent<DecisionRequester>().DecisionPeriod,
                ["decision_step"] = behavior.GetComponent<DecisionRequester>().DecisionStep,
                ["take_actions_between_decisions"] = behavior.GetComponent<DecisionRequester>().TakeActionsBetweenDecisions,
            };

            string json = JsonConvert.SerializeObject(features);
            _channel.SendString(json);
        }

        void OnDestroy()
        {
            Application.logMessageReceived -= _channel.SendDebugStatementToPython;
            if (Academy.IsInitialized)
            {
                SideChannelManager.UnregisterSideChannel(_channel);
            }
        }
    }
}
