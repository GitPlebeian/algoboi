//
//  ChartManager.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 11/12/23.
//

import Cocoa

class ChartManager {
    
    static let shared = ChartManager()
    
    weak var currentStockView: StockView?
    
    var currentAggregate: StockAggregate?
    var indicatorData: IndicatorData?
    
    func currentlyChartingStock() -> Bool {
        return currentAggregate != nil
    }
    
    func chartStock(_ aggregate: StockAggregate) {
        
        self.currentAggregate = aggregate
        currentStockView?.stockAggregate = aggregate
    }
    
    func setIndicatorData(_ indicatorData: IndicatorData) {
        self.indicatorData = indicatorData
        currentStockView?.addColoredLines(lines: [(indicatorData.ema14, CGColor.ChartEMALines.NSColor())])
        currentStockView?.addColoredLines(lines: [(indicatorData.ema28, CGColor.ChartEMALines.NSColor())])
        currentStockView?.addColoredLines(lines: [(indicatorData.sma50, CGColor.LightSMALine.NSColor())])
        currentStockView?.addColoredLines(lines: [(indicatorData.sma200, CGColor.DarkSMALine.NSColor())])
        currentStockView?.mouseDelegate = self
    }
    
    func setAuxSets(auxSets: [StockViewAuxGraphProperties]) {
        currentStockView?.setAuxViews(auxSets)
    }
}

extension ChartManager: StockViewMouseDelegate {
    func candleHoverChanged(index: Int) {
        guard let data = indicatorData else {return}
        guard let aggregate = currentAggregate else {return}
        if index < 0 || index >= data.length {return}
        
        LabelValueController.shared.setLabelValue(index: 0, label: "1", value: "\(data.predicted1CandleOut[index].toRoundedString(precision: 2))")
        LabelValueController.shared.setLabelValue(index: 3, label: "2", value: "\(data.predicted2CandleOut[index].toRoundedString(precision: 2))")
        LabelValueController.shared.setLabelValue(index: 6, label: "3", value: "\(data.predicted3CandleOut[index].toRoundedString(precision: 2))")
        LabelValueController.shared.setLabelValue(index: 9, label: "4", value: "\(data.predicted4CandleOut[index].toRoundedString(precision: 2))")
        LabelValueController.shared.setLabelValue(index: 12, label: "5", value: "\(data.predicted5CandleOut[index].toRoundedString(precision: 2))")
    }
}
