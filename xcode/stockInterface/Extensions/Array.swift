//
//  Array.swift
//  stockInterface
//
//  Created by CHONK on 6/27/25.
//

import Foundation

extension Array {
    /// Returns `k` random elements from the array (or the whole array if k ≥ count).
    func randomSample(_ k: Int) -> [Element] {
        guard k > 0, !isEmpty else { return [] }
        let n = Swift.min(k, count)
        return Array(shuffled().prefix(n))
    }

    /// Returns a random sample of the array based on `fraction` (0.0…1.0).
    /// e.g. 1.0 → all elements; 0.5 → half the elements (rounded to nearest).
    func randomSample(fraction: Double) -> [Element] {
        guard fraction > 0, !isEmpty else { return [] }
        // clamp fraction between 0 and 1
        let clamped = Swift.max(0.0, Swift.min(1.0, fraction))
        let n = Int((Double(count) * clamped).rounded())
        return randomSample(n)
    }
}
