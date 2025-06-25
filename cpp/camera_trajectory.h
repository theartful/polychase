// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "pnp/types.h"
#include "utils.h"

class CameraTrajectory {
   public:
    CameraTrajectory() : first_frame_id(0) {}

    CameraTrajectory(int32_t first_frame_id, size_t count)
        : states(count), first_frame_id(first_frame_id) {}

    CameraTrajectory(CameraTrajectory&& other)
        : states(std::exchange(other.states, {})),
          first_frame_id(other.first_frame_id) {}

    CameraTrajectory(const CameraTrajectory& other)
        : states(other.states), first_frame_id(other.first_frame_id) {}

    CameraTrajectory& operator=(CameraTrajectory&& other) {
        states = std::exchange(other.states, {});
        first_frame_id = other.first_frame_id;

        return *this;
    }

    CameraTrajectory& operator=(const CameraTrajectory& other) {
        states = other.states;
        first_frame_id = other.first_frame_id;

        return *this;
    }

    bool IsValidFrame(int32_t frame_id) const {
        return Index(frame_id) < Count();
    }

    bool IsFrameFilled(int32_t frame_id) const {
        return IsValidFrame(frame_id) && Get(frame_id).has_value();
    }

    const std::optional<CameraState>& Get(int32_t frame_id) const {
        const size_t index = Index(frame_id);
        CHECK(index < Count());

        return states[index];
    }

    std::optional<CameraState>& GetMutable(int32_t frame_id) {
        const size_t index = Index(frame_id);
        CHECK(index < Count());

        return states[index];
    }

    void Set(int32_t frame_id, const CameraState& state) {
        const size_t index = Index(frame_id);
        CHECK(index < Count());

        states[index] = state;
    }

    void Clear(int32_t frame_id) {
        const size_t index = Index(frame_id);
        CHECK(index < Count());

        states[index] = std::nullopt;
    }

    size_t Count() const { return states.size(); }

    int32_t FirstFrame() const { return first_frame_id; }

    int32_t LastFrame() const { return first_frame_id + states.size() - 1; }

    size_t Index(int32_t frame_id) const {
        return static_cast<size_t>(frame_id - first_frame_id);
    }

   private:
    std::vector<std::optional<CameraState>> states;
    int32_t first_frame_id = 0;
};
