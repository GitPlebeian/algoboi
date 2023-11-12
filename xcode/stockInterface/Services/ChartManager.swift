//
//  ChartManager.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 11/12/23.
//

import Foundation

class ChartManager {
    
    static let shared = ChartManager()
    
    weak var currentStockView: StockView?
    
    var currentAggregate: StockAggregate?
    
    func chartStock(_ aggregate: StockAggregate) {
        
        self.currentAggregate = aggregate
        currentStockView?.stockAggregate = aggregate
    }
}
