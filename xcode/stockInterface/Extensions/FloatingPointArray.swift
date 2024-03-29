//
//  Array.swift
//  stockInterface
//
//  Created by CHONK on 10/17/23.
//

import Foundation

extension Array where Element: FloatingPoint {

    func sum() -> Element {
        return self.reduce(0, +)
    }

    func avg() -> Element {
        return self.sum() / Element(self.count)
    }

    func std() -> Element {
        let mean = self.avg()
        let v = self.reduce(0, { $0 + ($1-mean)*($1-mean) })
        return sqrt(v / (Element(self.count) - 1))
    }
    
    func sma() -> Element {
        let total = self.reduce(0, {$0 + $1})
        return total / Element(self.count)
    }
    
    func percentageChanges(compare: [Element]) -> [Element] {
        var results: [Element] = []
        if compare.count != self.count {return []}
        for (i, startingValue) in self.enumerated() {
            if startingValue == 0 {
                results.append(0)
                continue
            }
            results.append((compare[i] - startingValue) / abs(startingValue))
//            results.append(startingValue)
        }
        return results
    }
    
    func minus(compare: [Element]) -> [Element] {
        var results: [Element] = []
        if compare.count != self.count {return []}
        for (i, starting) in self.enumerated() {
            results.append(starting - compare[i])
        }
        return results
    }
}
