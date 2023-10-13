//
//  LayoutManager.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 10/13/23.
//

import Cocoa

class LayoutManager {
    
    static let shared = LayoutManager()
    
    // MARK: Properties
    
    var celledMasterView: CelledMasterView!
    
    private let numCellsWide: Int = 12
    private var numCellsHigh: Int = 0
    private var cellWidth: CGFloat = 0
    private var cellHeight: CGFloat = 0
    
    // MARK: Init
    
    init() {
        self.celledMasterView = CelledMasterView()
    }
    
    // MARK: Getters Setters
    
    // Used when initializing window
    func getCelledMasterView() -> NSView { return celledMasterView }
    
    // MARK: TBD
    
    func didUpdateMasterViewFrame(newFrame frame: NSRect) {
        calculateCellDimensions(frame: frame)
    }
    
    // MARK: Private
    
    private func calculateCellDimensions(frame: CGRect) {
        
        cellWidth = frame.width / 12
        var closestDifference = CGFloat.greatestFiniteMagnitude
        var bestCellHeight: CGFloat = 0
        for potentialCellsInHeight in 1...Int(frame.height) {
            let potentialCellHeight = frame.height / CGFloat(potentialCellsInHeight)
            let difference = abs(potentialCellHeight - cellWidth)
            if difference < closestDifference {
                closestDifference = difference
                bestCellHeight = potentialCellHeight
                self.numCellsHigh = potentialCellsInHeight
            }
            if difference == 0 || potentialCellHeight < cellWidth {
                break
            }
        }
        cellHeight = bestCellHeight
    }

}
